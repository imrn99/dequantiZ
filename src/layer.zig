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

// fn buildLayerMetaData(
//     allocator: std.mem.Allocator,
//     header: []const u8,
// ) [16]u8 {}

const LayerMetaData = struct {
    input_layernorm_weight: BlockMetaData(1),
    mlp: struct {
        down_p: TensorMetaData,
        gate_p: TensorMetaData,
        up_p: TensorMetaData,
    },
    post_att_layernorm_weight: BlockMetaData(1),
    att: struct {
        k_p: TensorMetaData,
        o_p: TensorMetaData,
        q_p: TensorMetaData,
        v_p: TensorMetaData,
    },
};

// tensor

const TensorInitError = error{ MissingKey, MalformedBlock };

const TensorMetaData = struct {
    input_scale: BlockMetaData(0),
    weight: BlockMetaData(2),
    group_scale: BlockMetaData(2),
    global_scale: BlockMetaData(0),

    fn buildTensorMD(
        layer_id: u4,
        tensor_name: []const u8,
        allocator: std.mem.Allocator,
        header: []const u8,
    ) !@This() {
        const key_input_scale = try std.fmt.allocPrint(
            allocator,
            "layers.{}.{s}.input_scale\"", // \" is required to differentiate weight/weight_scale/weight_scale_2
            .{ layer_id, tensor_name },
        );
        defer allocator.free(key_input_scale);
        const input_scale_sub = try findJSONSubString(key_input_scale, header);
        const input_scale: BlockMetaData(0) = try .parseFromSlice(
            allocator,
            header[input_scale_sub.start..input_scale_sub.end],
        );

        const key_weight = try std.fmt.allocPrint(
            allocator,
            "layers.{}.{s}.weight\"",
            .{ layer_id, tensor_name },
        );
        defer allocator.free(key_weight);
        const weight_sub = try findJSONSubString(key_weight, header);
        const weight: BlockMetaData(2) = try .parseFromSlice(
            allocator,
            header[weight_sub.start..weight_sub.end],
        );

        const key_group_scale = try std.fmt.allocPrint(
            allocator,
            "layers.{}.{s}.weight_scale\"",
            .{ layer_id, tensor_name },
        );
        defer allocator.free(key_group_scale);
        const group_scale_sub = try findJSONSubString(key_group_scale, header);
        const group_scale: BlockMetaData(2) = try .parseFromSlice(
            allocator,
            header[group_scale_sub.start..group_scale_sub.end],
        );

        const key_global_scale = try std.fmt.allocPrint(
            allocator,
            "layers.{}.{s}.weight_scale_2\"",
            .{ layer_id, tensor_name },
        );
        defer allocator.free(key_global_scale);
        const global_scale_sub = try findJSONSubString(key_global_scale, header);
        const global_scale: BlockMetaData(0) = try .parseFromSlice(
            allocator,
            header[global_scale_sub.start..global_scale_sub.end],
        );

        return .{
            .input_scale = input_scale,
            .weight = weight,
            .group_scale = group_scale,
            .global_scale = global_scale,
        };
    }
};

fn findJSONSubString(
    key: []const u8,
    header: []const u8,
) !struct { start: usize, end: usize } {
    const start = std.mem.find(u8, header, key) orelse {
        return TensorInitError.MissingKey;
    };
    const end = std.mem.findPos(u8, header, start, "}") orelse {
        return TensorInitError.MalformedBlock;
    };

    // added offset make the slice start and end on brackets of the block
    return .{ .start = start + key.len + 1, .end = end + 1 };
}

// individual data block

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
        ) !@This() {
            const raw = std.json.parseFromSlice(RawBlockMetaData(DIM), allocator, s, .{}) catch {
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
    const parsed = try BlockMetaData(0).parseFromSlice(gpa, in);
    try std.testing.expect(parsed.dtype == DType.fl32);
    try std.testing.expect(std.mem.eql(usize, &parsed.shape, &[0]usize{}));
    try std.testing.expect(parsed.offsets.start == 892);
    try std.testing.expect(parsed.offsets.end == 896);
}
test "dim1 block metadata deser" {
    const gpa = std.testing.allocator;
    const in = "{\"dtype\":\"BF16\",\"shape\":[2048],\"data_offsets\":[536986496,536990592]}";
    const parsed = try BlockMetaData(1).parseFromSlice(gpa, in);
    try std.testing.expect(parsed.dtype == DType.bfl16);
    try std.testing.expect(std.mem.eql(usize, &parsed.shape, &[1]usize{2048}));
    try std.testing.expect(parsed.offsets.start == 536986496);
    try std.testing.expect(parsed.offsets.end == 536990592);
}
test "dim2 block metadata deser" {
    const gpa = std.testing.allocator;
    const in = "{\"dtype\":\"F8_E4M3\",\"shape\":[2048,192],\"data_offsets\":[587207552,587600768]}";
    const parsed = try BlockMetaData(2).parseFromSlice(gpa, in);
    try std.testing.expect(parsed.dtype == DType.fl8_e4m3);
    try std.testing.expect(std.mem.eql(usize, &parsed.shape, &[2]usize{ 2048, 192 }));
    try std.testing.expect(parsed.offsets.start == 587207552);
    try std.testing.expect(parsed.offsets.end == 587600768);
}
test "tensor metadata deser" {
    const gpa = std.testing.allocator;
    // gibberish input, looking for "layers.9.mlp.down_proj.*"
    const in = "\"layers.8.self_attn.v_proj.input_scale\":{\"dtype\":\"F32\",\"shape\":[],\"data_offsets\":[832,836]},\"layers.8.self_attn.v_proj.weight_scale_2\":{\"dtype\":\"F32\",\"shape\":[],\"data_offsets\":[836,840]},\"layers.9.mlp.down_proj.input_scale\":{\"dtype\":\"F32\",\"shape\":[],\"data_offsets\":[840,844]},\"layers.9.mlp.down_proj.weight_scale_2\":{\"dtype\":\"F32\",\"shape\":[],\"data_offsets\":[844,848]},\"layers.9.mlp.gate_proj.input_scale\":{\"dtype\":\"F32\",\"shape\":[],\"data_offsets\":[848,852]},o_proj.weight\":{\"dtype\":\"U8\",\"shape\":[2048,1536],\"data_offsets\":[993137536,996283264]},\"layers.8.self_attn.q_proj.weight\":{\"dtype\":\"U8\",\"shape\":[3072,1024],\"data_offsets\":[996283264,999428992]},\"layers.8.self_attn.v_proj.weight\":{\"dtype\":\"U8\",\"shape\":[1024,1024],\"data_offsets\":[999428992,1000477568]},\"layers.9.mlp.down_proj.weight\":{\"dtype\":\"U8\",\"shape\":[2048,3072],\"data_offsets\":[1000477568,1006769024]},\"layers.9.mlp.gate_proj.weight\":{\"dtype\":\"U8\",\"shape\":[6144,1024],\"data_offsets\":[1006769024,1013060480]},\"layers.8.self_attn.v_proj.weight_scale\":{\"dtype\":\"F8_E4M3\",\"shape\":[1024,128],\"data_offsets\":[587993984,588125056]},\"layers.9.mlp.down_proj.weight_scale\":{\"dtype\":\"F8_E4M3\",\"shape\":[2048,384],\"data_offsets\":[588125056,588911488]},\"layers.9.mlp.gate_proj.weight_scale\":{\"dtype\":\"F8_E4M3\",\"shape\":[6144,128],\"data_offsets\":[588911488,589697920]},\"layers.9.mlp.up_proj.weight_scale\":{\"dtype\":\"F8_E4M3\",\"shape\":[6144,128],\"data_offsets\":[589697920,590484352]},\"layers.9.self_attn.k_proj.weight_scale\":{\"dtype\":\"F8_E4M3\",\"shape\":[1024,128],\"data_offsets\":[590484352,590615424]}";
    const parsed = try TensorMetaData.buildTensorMD(9, "mlp.down_proj", gpa, in);
    // try std.testing.expect(parsed.weight.dtype == DType.fl8_e4m3);
    _ = parsed;
}
