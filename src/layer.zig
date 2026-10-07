const std = @import("std");

pub const DType = enum {
    bfl16,
    fl32,
    un8,
    fl8_e4m3,

    fn sizeOf(self: *const @This()) usize {
        switch (self) {
            .bfl16 => {
                return 2;
            },
            .fl32 => {
                return 4;
            },
            .un8 => {
                return 1;
            },
            .fl8_e4m3 => {
                return 1;
            },
        }
    }
};

pub const BlockInitError = error{ FailedDeser, UnknownDType };

pub fn BlockMetaData(DIM: comptime_int) type {
    return struct {
        dtype: DType,
        shape: [DIM]usize,
        offsets: struct { start: usize, end: usize },

        // probably a bad interface, it takes an allocator but
        // the resulting struct doesn't require deinit call
        pub fn parseFromSlice(
            allocator: std.mem.Allocator,
            s: []const u8,
            options: std.json.ParseOptions,
        ) !@This() {
            const raw = std.json.parseFromSlice(RawBlockMetaData(DIM), allocator, s, options) catch {
                return BlockInitError.FailedDeser;
            };
            defer raw.deinit();
            const md = try @This().fromRaw(raw.value);
            return md;
        }

        fn fromRaw(in: RawBlockMetaData(DIM)) !@This() {
            // ?
            const dtype: DType = if (std.mem.eql(u8, in.dtype, "BF16"))
                DType.bfl16
            else if (std.mem.eql(u8, in.dtype, "F32"))
                DType.fl32
            else if (std.mem.eql(u8, in.dtype, "U8"))
                DType.un8
            else if (std.mem.eql(u8, in.dtype, "F8_E4M3"))
                DType.fl8_e4m3
            else
                return BlockInitError.UnknownDType;

            return .{ .dtype = dtype, .shape = in.shape, .offsets = .{
                .start = in.data_offsets[0],
                .end = in.data_offsets[1],
            } };
        }
    };
}

fn RawBlockMetaData(DIM: comptime_int) type {
    return struct {
        dtype: []u8,
        shape: [DIM]usize,
        data_offsets: [2]usize,
    };
}

test "dim0 block metadata deser" {
    const gpa = std.testing.allocator;
    const in = "{\"dtype\":\"F32\",\"shape\":[],\"data_offsets\":[892,896]}";
    const parsed = try BlockMetaData(0).parseFromSlice(gpa, in, .{});
    try std.testing.expect(parsed.dtype == DType.fl32);
    try std.testing.expect(std.mem.eql(usize, &parsed.shape, &[0]usize{}));
    try std.testing.expect(parsed.offsets.start == 892);
    try std.testing.expect(parsed.offsets.end == 896);
}
test "dim1 block metadata deser" {
    const gpa = std.testing.allocator;
    const in = "{\"dtype\":\"BF16\",\"shape\":[2048],\"data_offsets\":[536986496,536990592]}";
    const parsed = try BlockMetaData(1).parseFromSlice(gpa, in, .{});
    try std.testing.expect(parsed.dtype == DType.bfl16);
    try std.testing.expect(std.mem.eql(usize, &parsed.shape, &[1]usize{2048}));
    try std.testing.expect(parsed.offsets.start == 536986496);
    try std.testing.expect(parsed.offsets.end == 536990592);
}
test "dim2 block metadata deser" {
    const gpa = std.testing.allocator;
    const in = "{\"dtype\":\"F8_E4M3\",\"shape\":[2048,192],\"data_offsets\":[587207552,587600768]}";
    const parsed = try BlockMetaData(2).parseFromSlice(gpa, in, .{});
    try std.testing.expect(parsed.dtype == DType.fl8_e4m3);
    try std.testing.expect(std.mem.eql(usize, &parsed.shape, &[2]usize{ 2048, 192 }));
    try std.testing.expect(parsed.offsets.start == 587207552);
    try std.testing.expect(parsed.offsets.end == 587600768);
}
