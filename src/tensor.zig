const std = @import("std");
const builtin = @import("builtin");
const linux = std.os.linux;
const Io = std.Io;

const metadata = @import("metadata.zig");
const decode = @import("decode.zig");

pub const BFloat16 = struct {
    bits: u16,

    pub fn toU16(this: @This()) u16 {
        return this.bits;
    }
};

const TensorBuildingError = error{ FailedGlobalScaleDeser, FailedGroupScaleDeser, FailedWeightDeser };

// NOTE: no complex tiling pattern needed bc group scales are defined
//       on groups along the stride==1 axis and we do CPU exec
pub fn buildTensorData(
    io: Io,
    allocator: std.mem.Allocator,
    st_file: *const Io.File,
    n_threads: usize,
    tensor_md: metadata.TensorMetaData,
) ![]BFloat16 {
    // try pinCurrentThread(0);
    const now = std.Io.Clock.Timestamp.now(io, .real);

    const n_elem = tensor_md.weight.shape[0] * tensor_md.weight.shape[1] * 2; // two elements per u8 value
    const n_groups = n_elem / 16;
    const out: []align(32) BFloat16 = try allocator.alignedAlloc(
        BFloat16,
        std.mem.Alignment.@"32",
        n_elem,
    );
    // @memset(out, .{ .bits = 0 }); // !

    // GLOBAL SCALE

    var tmp: [4]u8 = .{ 0, 0, 0, 0 };
    const n = st_file.readPositionalAll(io, &tmp, tensor_md.global_scale.offsets.start) catch |e| {
        std.log.err("{}", .{e});
        return TensorBuildingError.FailedGlobalScaleDeser;
    };
    if (n != 4) {
        return TensorBuildingError.FailedGlobalScaleDeser;
    }
    const global_scale: f32 = @bitCast(tmp);

    // GROUP SCALES

    const group_in: []u8 = try allocator.alloc(u8, n_groups);
    defer allocator.free(group_in);
    const ng = st_file.readPositionalAll(io, group_in, tensor_md.group_scale.offsets.start) catch |e| {
        std.log.err("{}", .{e});
        return TensorBuildingError.FailedGroupScaleDeser;
    };
    if (ng != n_groups) {
        std.log.err("could not read group scale value", .{});
        return TensorBuildingError.FailedGroupScaleDeser;
    }

    // WEIGHTS

    const weight_in: []u8 = try allocator.alloc(u8, n_groups * 8);
    defer allocator.free(weight_in);
    const nw = st_file.readPositionalAll(io, weight_in, tensor_md.weight.offsets.start) catch |e| {
        std.log.err("{}", .{e});
        return TensorBuildingError.FailedWeightDeser;
    };
    if (nw != n_groups * 8) {
        std.log.err("could not read weight values", .{});
        return TensorBuildingError.FailedWeightDeser;
    }

    std.log.info(
        "data file read time: {f}",
        .{now.durationTo(std.Io.Clock.Timestamp.now(io, .real)).raw},
    );

    // wrok split & dispatch
    // for now, assume everything has "nice" dimensions
    // may have to change if I execute this on a Grace bc of the bs number of procs

    var out_shards: std.ArrayList([]BFloat16) = try .initCapacity(allocator, n_threads);
    var handles: std.ArrayList(Io.Future(void)) = try .initCapacity(allocator, n_threads);
    defer handles.deinit(allocator);

    const n_groups_per_thread = n_groups / n_threads;

    for (0..n_threads) |i| {
        // io.async could be sufficient here? io API is extremely unclear
        // the best doc it has is from blog posts, explicit threading might be better atp
        const shard: []align(32) BFloat16 = try allocator.alignedAlloc(
            BFloat16,
            std.mem.Alignment.@"32",
            n_groups_per_thread * 16,
        );
        const fut = try io.concurrent(readDecodeWrite, .{
            io,
            i,
            n_groups_per_thread,
            weight_in,
            group_in,
            global_scale,
            shard,
        });
        try out_shards.append(allocator, shard);
        try handles.append(allocator, fut);
    }

    for (handles.items) |h| {
        const tp: *Io.Future(void) = @constCast(&h);
        tp.await(io);
    }

    for (out_shards.items, 0..) |s, i| {
        @memcpy(out[i * s.len ..(i+1) * s.len ], s);
        allocator.free(s);
    }

    std.log.info(
        "total tensor data init time: {f}",
        .{now.durationTo(std.Io.Clock.Timestamp.now(io, .real)).raw},
    );

    return out;
}

// 1st level: threads
fn readDecodeWrite(
    io: Io,
    thread_id: usize,
    n_groups: usize,
    weights_raw: []const u8,
    group_scales_raw: []const u8,
    global_scale: f32,
    out: []align(32) BFloat16,
) void {
    // printAffinity();
    pinCurrentThread(thread_id) catch |e| {
        std.log.err("couldn't set affinity: {}", .{e});
    };
    // printAffinity();
    // const out_start = thread_id * n_groups * 16;
    // @memset(out[out_start..out_start + n_groups * 16], .{ .bits = 0 });
    @memset(out, .{ .bits = 0 });

    const now = std.Io.Clock.Timestamp.now(io, .real);
    // 2nd level: groups
    // the code assume slice dimensions are correct
    // casting the slice below directly as [][16]u4 yields worse perf than this + bitcast later
    const grouped_weights_raw: []const [8]u8 = std.mem.bytesAsSlice([8]u8, weights_raw);
    // _ = weights_raw;
    // _ = group_scales_raw;
    // _ = global_scale;

    // w: 8 * u8 = 16 * encoded fp4 E2M1
    // g: 1 * u8 = 1 * fp8 E4M3
    for (0..n_groups) |i| {
        // 3rd level: vectors
        // from zig's doc, I'm assuming processing the entire thing
        // in a single 16-wide vec compiles to the most optimal work split
        // NOTE: - LUT impl result in scalar loads, but still outperforms the vectorized decode
        //       - overhead of the direct bit manipulation decode seems to come mostly from the
        //         extension to u8; similar overhead appears when using manually unpacked u8
        //         with LUT lookup

        // LUT-based decode

        const encoded_weights: @Vector(16, u4) = @bitCast(grouped_weights_raw[i]);
        var decoded_weights: @Vector(16, f32) = undefined;
        inline for (0..16) |j| {
            decoded_weights[j] = decode.E2M1_LUT[@intCast(encoded_weights[j])];
        }

        // bit manipulation decode -- slower

        // const encoded_weights: @Vector(8, u8) = @bitCast(grouped_weights_raw[i]);
        // const hi = encoded_weights >> @splat(4);
        // const lo = encoded_weights & @as(@Vector(8, u8), @splat(0b00001111));
        // const encoded_weights_extended: @Vector(16, u8) = @shuffle(
        //     u8,
        //     hi,
        //     lo,
        //    @Vector(16, i32){ 0, -1, 1, -2, 2, -3, 3, -4, 4, -5, 5, -6, 6, -7, 7, -8},
        // );
        // var decoded_weights: @Vector(16, f32) = decode.decodeE2M1(encoded_weights_extended);

        // apply scales and write

        const decoded_gscale: f32 = decode.E4M3_LUT[@intCast(group_scales_raw[i])];
        decoded_weights *= @splat(decoded_gscale);
        decoded_weights *= @splat(global_scale);

        // apparently f32 -> bf16 is platform dependent? for now, I'll truncate
        // https://en.wikipedia.org/wiki/Bfloat16_floating-point_format#Rounding_and_conversion
        // NOTE: rounding and truncating have significantly different costs
        const bf16_values: @Vector(16, u16) = @truncate(@as(
            @Vector(16, u32),
            @bitCast(decoded_weights),
        ) >> @splat(16));
        const out_group_start = i * 16;

        writePayload(@ptrCast(@alignCast(out[out_group_start .. out_group_start + 16])), bf16_values);

        // NOTE: - commenting out the decoding logic gives pure write throughput,
        //         replacing the write code above with a memset yields the same timing
        //       - threading scales poorly in both situation on the Intel CPU, but that could
        //         very well be due to (lack of) pinning, and P/E core action
        //       - the pure write takes
        //         - ~66% of the time of the full regular routine on Intel CPU,
        //         - the same time on M1 Pro
        // @memset(out[out_group_start..out_group_start + 16], .{ .bits = 166} );
    }

    std.log.info(
        "┗━thread #{} decode+write: {f}",
        .{ thread_id, now.durationTo(std.Io.Clock.Timestamp.now(io, .real)).raw },
    );
}

inline fn writePayload(
    dst: *align(32) @Vector(16, u16),
    value: @Vector(16, u16),
) void {
    if (builtin.cpu.arch == std.Target.Cpu.Arch.x86_64 and
        std.Target.x86.featureSetHas(builtin.cpu.features, std.Target.x86.Feature.avx))
    {
        asm volatile (
            \\vmovntdq %[value], (%[dst])
            :
            : [value] "x" (value),
              [dst] "r" (dst),
            : .{ .memory = true });
    } else if (builtin.cpu.arch == std.Target.Cpu.Arch.aarch64 and
        std.Target.x86.featureSetHas(builtin.cpu.features, std.Target.aarch64.Feature.neon))
    {
        // TODO: impl ARM inline asm; STNP instr?
        // https://www.scs.stanford.edu/~zyedidia/arm64/stnp_fpsimd.html
        return;
    } else {
        dst.* = value;
    }
}

fn pinCurrentThread(core_id: usize) !void {
    comptime if (builtin.target.os.tag != std.Target.Os.Tag.linux) {
        return error.UnsupportedOS;
    };

    var mask: linux.cpu_set_t = undefined;
    @memset(&mask, 0);

    // arithmetic is necessary to handle IDs >64
    if (core_id >= @bitSizeOf(linux.cpu_set_t))
        return error.InvalidCoreID;

    const word_bits = @bitSizeOf(usize);
    mask[core_id / word_bits] = @as(usize, 1) << @intCast(core_id % word_bits);

    try linux.sched_setaffinity(0, &mask);
}

fn printAffinity() void {
    const aff = std.posix.sched_getaffinity(0) catch |e| {
        std.log.err("couldn't fetch current process affinity: {}", .{e});
        return;
    };
    std.log.info("affinity: 0b{b}", .{aff[0]});
    // comment above line & uncomment the for loop for machines with >64 logic cores
    // for (aff) |a| {
    //     std.log.info("affinity: 0b{b}", .{a});
    // }
}
