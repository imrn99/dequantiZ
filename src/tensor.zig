const std = @import("std");
const Io = std.Io;

const metadata = @import("metadata.zig");

pub const BFloat16 = struct {
    sign: u1,
    exponent: u8,
    mantissa: u7,

    // apparently this is platform dependent? for now, I'll truncate
    // https://en.wikipedia.org/wiki/Bfloat16_floating-point_format#Rounding_and_conversion
    fn fromF32(in: f32) @This() {
        const bits: [32]u1 = @bitCast(in);

        return .{
            .sign = bits[31],
            .exponent = @bitCast(bits[23..31]),
            .mantissa = @bitCast(bits[16..23]),
        };
    }
};

// concurrent file read with zig IO
// each thread works on its own input and output slices.
// no complex tiling pattern needed bc
// group scales are defined on groups along the stride==1 axis and we do CPU exec
fn buildTensor(
    io: Io,
    allocator: std.mem.Allocator,
    st_file: *const Io.File,
    n_threads: usize,
    tensor_md: metadata.TensorMetaData,
) !std.ArrayList(BFloat16) {
    const n_elem = tensor_md.weight.shape[0] * tensor_md.weight.shape[1] * 2; // two elements per u8 value
    const out: std.ArrayList(BFloat16) = .initCapacity(allocator, n_elem);
    var handles: std.ArrayList(Io.Future(void)) = .initCapacity(allocator, n_threads);

    
    for (0..n_threads) |_| {
        // io.async could be sufficient here?
        const fut = try io.concurrent(readDecodeWrite, .{st_file});
        try handles.append(allocator, fut);
    }

    for (handles) |h| {
        try h.await(io);
    }

    return out;
}

fn readDecodeWrite(
    io: Io,
    allocator: std.mem.Allocator,
    st_file: *const Io.File,
) void {
    _ = io;
    _ = allocator;
    _ = st_file;
}
