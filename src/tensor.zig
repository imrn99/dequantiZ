const std = @import("std");
const Io = std.Io;

const metadata = @import("metadata.zig");
const decode = @import("decode.zig");

pub const BFloat16 = struct {
    bits: u16,

    // apparently this is platform dependent? for now, I'll truncate
    // https://en.wikipedia.org/wiki/Bfloat16_floating-point_format#Rounding_and_conversion
    pub fn fromF32(value: f32) @This() {
        const f32_bits: u32 = @bitCast(value);

        return .{
            .bits = @truncate(f32_bits >> 16),
        };
    }

    pub fn toU16(this: @This()) u16 {
        return this.bits;
    }
};

const TensorBuildingError = error{ FailedGlobalScaleDeser, FailedWeightRead };

// NOTE: no complex tiling pattern needed bc group scales are defined
//       on groups along the stride==1 axis and we do CPU exec
pub fn buildTensorData(
    io: Io,
    allocator: std.mem.Allocator,
    st_file: *const Io.File,
    n_threads: usize,
    tensor_md: metadata.TensorMetaData,
) ![]BFloat16 {
    const n_elem = tensor_md.weight.shape[0] * tensor_md.weight.shape[1] * 2; // two elements per u8 value
    const out: []BFloat16 = try allocator.alloc(BFloat16, n_elem);
    var handles: std.ArrayList(Io.Future(void)) = try .initCapacity(allocator, n_threads);
    defer handles.deinit(allocator);

    const now = std.Io.Clock.Timestamp.now(io, .real);

    var tmp: [4]u8 = .{ 0, 0, 0, 0 };
    const n = st_file.readPositionalAll(io, &tmp, tensor_md.global_scale.offsets.start) catch |e| {
        std.log.err("{}", .{e});
        return TensorBuildingError.FailedGlobalScaleDeser;
    };
    if (n != 4) {
        return TensorBuildingError.FailedGlobalScaleDeser;
    }
    const global_scale: f32 = @bitCast(tmp);

    // group sizing
    // for now, assume everything has "nice" dimensions
    // may have to change if I execute this on a Grace bc of the bs number of procs

    // const rem = try std.math.rem(usize, n_elem, n_threads);
    // std.debug.assert(try std.math.rem(usize, n_elem, 16) != 0);
    // const n_groups = n_elem / 16;
    // var n_groups_per_thread
    const block_size = n_elem / n_threads;
    // if (try std.math.rem(usize, block_size, 16) != 0) {}
    for (0..n_threads) |i| {
        const start = tensor_md.weight.offsets.start + i * block_size;
        // var n_elem_to_process = block_size;
        // if (i == n_threads - 1 and rem != 0) n_elem_to_process += rem;

        // io.async could be sufficient here? io API is extremely unclear
        // the best doc it has is from blog posts, explicit threading might be better atp
        const fut = try io.concurrent(readDecodeWrite, .{
            io,
            allocator,
            st_file,
            i,
            block_size / 16,
            start,
            tensor_md.group_scale.offsets.start,
            global_scale,
            out,
        });
        try handles.append(allocator, fut);
    }

    for (handles.items) |h| {
        const tp: *Io.Future(void) = @constCast(&h);
        tp.await(io);
    }

    const end = std.Io.Clock.Timestamp.now(io, .real);

    const elapsed = now.durationTo(end);
    std.log.info("time elapsed: {f}", .{elapsed.raw});

    return out;
}

// 1st level: threads
fn readDecodeWrite(
    io: Io,
    allocator: std.mem.Allocator,
    st_file: *const Io.File,
    thread_id: usize,
    n_groups: usize,
    weights_start: usize,
    group_scales_start: usize,
    global_scale: f32,
    out: []BFloat16,
) void {
    const now = std.Io.Clock.Timestamp.now(io, .real);

    //NOTE: tradeoff of reading all data in bufs from the start
    //      vs comptime-sized buffers in the inner loop
    //      removing file read from the hot loop may be the better plan
    //      it also will allow local storage of the input value, so likely better cache usage
    const weight_in: []u8 = allocator.alloc(u8, n_groups * 8) catch |e| {
        std.log.err("could not allocate weight reading buffer in thread {}: {}", .{ thread_id, e });
        return;
    };
    const nw = st_file.readPositionalAll(io, weight_in, weights_start) catch |e| {
        std.log.err("{}", .{e});
        return;
    };
    if (nw != n_groups * 8) {
        std.log.err("could not read weight values", .{});
        return;
    }

    const group_in: []u8 = allocator.alloc(u8, n_groups) catch |e| {
        std.log.err("could not allocate weight reading buffer in thread {}: {}", .{ thread_id, e });
        return;
    };
    const ng = st_file.readPositionalAll(io, group_in, group_scales_start) catch |e| {
        std.log.err("{}", .{e});
        return;
    };
    if (ng != n_groups) {
        std.log.err("could not read group scale value", .{});
        return;
    }

    // 2nd level: groups
    // the code assume slice dimensions are correct
    const grouped_weights: [][8]u8 = std.mem.bytesAsSlice([8]u8, weight_in);
    const out_start = thread_id * n_groups * 16;

    // w: 8 * u8 = 16 * encoded fp4 E2M1
    // g: 1 * u8 = 1 * fp8 E4M3
    for (grouped_weights, group_in, 0..) |w, g, i| {
        // 3rd level: vectors
        const encoded_weights: [16]u4 = @bitCast(w);
        const decoded_gscale: f32 = decode.E4M3_LUT[@intCast(g)];

        var lut: @Vector(16, usize) = @splat(@intFromPtr(&decode.E2M1_LUT));
        const offsets: @Vector(16, u4) = encoded_weights;
        const indices: @Vector(16, usize) = @intCast(offsets);
        lut += indices * @as(@Vector(16, usize), @splat(@sizeOf(f32)));

        var decoded_weights: @Vector(16, f32) = undefined;
        inline for (0..16) |j| {
            // decoded_weights[j] = decode.E2M1_LUT[@intCast(encoded_weights[j])];
            decoded_weights[j] = @as(*const f32, @ptrFromInt(lut[j])).*;
        }
        decoded_weights *= @splat(decoded_gscale);
        decoded_weights *= @splat(global_scale);

        // write result
        const out_group_start = out_start + i * 16;
        inline for (0..16) |j| {
            out[out_group_start + j] = BFloat16.fromF32(decoded_weights[j]);
        }
    }

    const end = std.Io.Clock.Timestamp.now(io, .real);

    const elapsed = now.durationTo(end);
    std.log.info("time elapsed in thread #{}: {f}", .{ thread_id, elapsed.raw });
}
