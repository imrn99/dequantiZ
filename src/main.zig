const std = @import("std");
const Io = std.Io;

const dequantiZ = @import("dequantiZ");

const TENSOR_FILE = "model.safetensors";
const LAYER: u4 = 0; // ref model has 16 layers, all with the same structure

pub fn main(init: std.process.Init) !void {
    const arena: std.mem.Allocator = init.arena.allocator();

    const io = init.io;
    const st_file = Io.failingDirOpenFile(null, Io.Dir.cwd(), "/" ++ TENSOR_FILE, .{}) catch |e|
        {
            std.log.err("while opening tensor file: {}", .{e});
            return;
        };
    const tmp: [8]u8 = .{ 0, 0, 0, 0, 0, 0, 0, 0 };
    const n = st_file.readPositionalAll(io, @constCast(&tmp), 0) catch |e| {
        std.log.err("{}", .{e});
        return;
    };
    if (n != 8) {
        std.log.err("{s}", .{"couldn't read header size"});
        return;
    }
    const header_size: u64 = @bitCast(tmp);
    var header_blob: std.ArrayList(u8) = std.ArrayList(u8).initCapacity(arena, @intCast(header_size)) catch |e| {
        std.log.err("{}", .{e});
        return;
    };
    defer header_blob.deinit(arena);
    const nn = st_file.readPositionalAll(io, @constCast(header_blob.items), 0) catch |e| {
        std.log.err("{}", .{e});
        return;
    };
    if (nn != header_blob.capacity) {
        std.log.err("{s}", .{"mismatch between read data and expected header length"});
        return;
    }
}
