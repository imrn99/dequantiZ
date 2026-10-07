const std = @import("std");
const Io = std.Io;

const dequantiZ = @import("dequantiZ");

const TENSOR_FILE = "model.safetensors";
const LAYER: u4 = 0; // ref model has 16 layers, all with the same structure

pub fn main(init: std.process.Init) !void {
    const arena: std.mem.Allocator = init.arena.allocator();

    const io = init.io;
    const st_file = Io.Dir.cwd().openFile(io, TENSOR_FILE, .{}) catch |e|
        {
            std.log.err("while opening tensor file: {}", .{e});
            return;
        };
    defer st_file.close(io);

    // header size
    var tmp: [8]u8 = .{ 0, 0, 0, 0, 0, 0, 0, 0 };
    const n = st_file.readPositionalAll(io, &tmp, 0) catch |e| {
        std.log.err("{}", .{e});
        return;
    };
    if (n != 8) {
        std.log.err("{s}", .{"couldn't read header size"});
        return;
    }
    const header_size: u64 = @bitCast(tmp);
    std.log.info("header size = {}", .{header_size});

    // header content
    var reader_buf: [1024]u8 = undefined;
    var reader = st_file.reader(io, &reader_buf);
    const header_blob = reader.interface.readAllocAll(arena, @intCast(header_size + 8)) catch |e| {
        std.log.err("{}", .{e});
        return;
    };
    if (header_blob.len != @as(usize, header_size + 8)) {
        std.log.err("...", .{});
        return;
    }

    // dump stdout to file for verification
    //var stdout_buffer: [1024]u8 = undefined;
    //var stdout_file_writer: Io.File.Writer = .init(.stdout(), io, &stdout_buffer);
    //try stdout_file_writer.interface.print("{s}\n", .{header_blob[8..]});
    //try stdout_file_writer.flush();
}
