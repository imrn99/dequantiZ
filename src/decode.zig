const std = @import("std");

// there is a classic compute vs mem tradeoff in the alg,
// we can export both look up tables and decoding functions to compare implementation down the line,
// look up tables are initialized using decode functions for some verification/redundancy

pub const E2M1_LUT = buildLookupTableE2M1();
pub const E4M3_LUT = buildLookupTableE4M3();

const V16u8 = @Vector(16, u8);
const V16u32 = @Vector(16, u32);

fn buildLookupTableE2M1() [16]f32 {
    var in: V16u8 = undefined;
    for (0..16) |i| {
        in[i] = i;
    }

    const out: [16]f32 = decodeE2M1(in);

    return out;
}

fn buildLookupTableE4M3() [256]f32 {
    var out: [256]f32 = undefined;
    @memset(&out, 0);

    for (0..256) |i| {
        out[i] = decodeE4M3(@intCast(i));
    }

    return out;
}

pub fn decodeE2M1(in: V16u8) @Vector(16, f32) {
    const s: V16u8 = in & @as(V16u8, @splat(0b00001000));
    const e: V16u8 = (in & @as(V16u8, @splat(0b00000110))) >> @splat(1);
    const m: V16u8 = in & @as(V16u8, @splat(0b00000001));

    // the way subnormal conversion works is:
    // if e_nvfp4 (2bits) == 0 => e_f32 (8bits) = 0
    //                    else => e_f32 (8bits) = 126 + e_nvfp4
    const non_zero_e: V16u8 = @intFromBool(e != @as(V16u8, @splat(0)));
    const non_zero_m: V16u8 = @intFromBool(m != @as(V16u8, @splat(0)));
    const adjusted_e: V16u8 = (non_zero_e | non_zero_m) * (e + @as(V16u8, @splat(126)));

    return @bitCast((@as(V16u32, @intCast(s)) << @splat(28)) |
        (@as(V16u32, @intCast(adjusted_e)) << @splat(23)) |
        (@as(V16u32, @intCast(non_zero_e * m)) << @splat(22)));
}

// I'm assuming the model uses the "finite" variant
// https://www.asawicki.info/articles/fp8_tables.php
pub fn decodeE4M3(in: u8) f32 {
    const bits: [8]u1 = @bitCast(in);
    const s = bits[7];
    const e: u4 = @bitCast(bits[3..7].*);
    const m: u3 = @bitCast(bits[0..3].*);

    if (e == 15 and m == 7) {
        return std.math.pow(f32, -1.0, @as(f32, @floatFromInt(s))) * std.math.nan(f32);
    }

    const non_zero_e: u1 = @bitCast(e != 0);
    const exp: u32 = @intCast(@max(e, 1));
    const sign: f32 = 1.0 - 2.0 * @as(f32, @floatFromInt(s));
    const two_pow: f32 = @bitCast((exp + 120) << 23);

    return sign * two_pow *
        (@as(f32, @floatFromInt(non_zero_e)) +
            @as(f32, @floatFromInt(m)) / 8.0);
}

test "e2m1_lut_values" {
    const ref: [16]f32 = .{
        0.0,  0.5,  1.0,  1.5,
        2.0,  3.0,  4.0,  6.0,
        -0.0, -0.5, -1.0, -1.5,
        -2.0, -3.0, -4.0, -6.0,
    };

    for (0..16) |i| {
        try std.testing.expectEqual(ref[i], E2M1_LUT[i]);
    }
}

test "e4m3_lut_values" {
    // LLM-generated from https://www.asawicki.info/articles/fp8_tables.php
    const ref: [256]f32 = .{
        0.0,       0.001953125,  0.00390625,  0.005859375,  0.0078125,  0.009765625,  0.01171875,  0.013671875,
        0.015625,  0.017578125,  0.01953125,  0.021484375,  0.0234375,  0.025390625,  0.02734375,  0.029296875,
        0.03125,   0.03515625,   0.0390625,   0.04296875,   0.046875,   0.05078125,   0.0546875,   0.05859375,
        0.0625,    0.0703125,    0.078125,    0.0859375,    0.09375,    0.1015625,    0.109375,    0.1171875,
        0.125,     0.140625,     0.15625,     0.171875,     0.1875,     0.203125,     0.21875,     0.234375,
        0.25,      0.28125,      0.3125,      0.34375,      0.375,      0.40625,      0.4375,      0.46875,
        0.5,       0.5625,       0.625,       0.6875,       0.75,       0.8125,       0.875,       0.9375,
        1.0,       1.125,        1.25,        1.375,        1.5,        1.625,        1.75,        1.875,
        2.0,       2.25,         2.5,         2.75,         3.0,        3.25,         3.5,         3.75,
        4.0,       4.5,          5.0,         5.5,          6.0,        6.5,          7.0,         7.5,
        8.0,       9.0,          10.0,        11.0,         12.0,       13.0,         14.0,        15.0,
        16.0,      18.0,         20.0,        22.0,         24.0,       26.0,         28.0,        30.0,
        32.0,      36.0,         40.0,        44.0,         48.0,       52.0,         56.0,        60.0,
        64.0,      72.0,         80.0,        88.0,         96.0,       104.0,        112.0,       120.0,
        128.0,     144.0,        160.0,       176.0,        192.0,      208.0,        224.0,       240.0,
        256.0,     288.0,        320.0,       352.0,        384.0,      416.0,        448.0,       std.math.nan(f32),
        -0.0,      -0.001953125, -0.00390625, -0.005859375, -0.0078125, -0.009765625, -0.01171875, -0.013671875,
        -0.015625, -0.017578125, -0.01953125, -0.021484375, -0.0234375, -0.025390625, -0.02734375, -0.029296875,
        -0.03125,  -0.03515625,  -0.0390625,  -0.04296875,  -0.046875,  -0.05078125,  -0.0546875,  -0.05859375,
        -0.0625,   -0.0703125,   -0.078125,   -0.0859375,   -0.09375,   -0.1015625,   -0.109375,   -0.1171875,
        -0.125,    -0.140625,    -0.15625,    -0.171875,    -0.1875,    -0.203125,    -0.21875,    -0.234375,
        -0.25,     -0.28125,     -0.3125,     -0.34375,     -0.375,     -0.40625,     -0.4375,     -0.46875,
        -0.5,      -0.5625,      -0.625,      -0.6875,      -0.75,      -0.8125,      -0.875,      -0.9375,
        -1.0,      -1.125,       -1.25,       -1.375,       -1.5,       -1.625,       -1.75,       -1.875,
        -2.0,      -2.25,        -2.5,        -2.75,        -3.0,       -3.25,        -3.5,        -3.75,
        -4.0,      -4.5,         -5.0,        -5.5,         -6.0,       -6.5,         -7.0,        -7.5,
        -8.0,      -9.0,         -10.0,       -11.0,        -12.0,      -13.0,        -14.0,       -15.0,
        -16.0,     -18.0,        -20.0,       -22.0,        -24.0,      -26.0,        -28.0,       -30.0,
        -32.0,     -36.0,        -40.0,       -44.0,        -48.0,      -52.0,        -56.0,       -60.0,
        -64.0,     -72.0,        -80.0,       -88.0,        -96.0,      -104.0,       -112.0,      -120.0,
        -128.0,    -144.0,       -160.0,      -176.0,       -192.0,     -208.0,       -224.0,      -240.0,
        -256.0,    -288.0,       -320.0,      -352.0,       -384.0,     -416.0,       -448.0,      -std.math.nan(f32),
    };

    for (0..256) |i| {
        if (i == 0x7F or i == 0xFF) continue; // skip nan values
        try std.testing.expectEqual(ref[i], E4M3_LUT[i]);
    }
}
