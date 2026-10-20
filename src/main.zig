const std = @import("std");
const Io = std.Io;

const dequantiZ = @import("dequantiZ");

const TENSOR_FILE = "model.safetensors";

// make this into CL arguments
const LAYER_ID: u4 = 0;
const N_THREADS: usize = 1;

pub fn main(init: std.process.Init) !void {
    const arena: std.mem.Allocator = init.arena.allocator();
    const io = init.io;
    const st_file = Io.Dir.cwd().openFile(io, TENSOR_FILE, .{}) catch |e|
        {
            std.log.err("while opening tensor file: {}", .{e});
            return;
        };
    defer st_file.close(io);

    const md = try parseHeader(io, arena, &st_file);
    const layer_md = md[@intCast(LAYER_ID)];

    const buf = try dequantiZ.tensor.buildTensorData(
        io,
        arena,
        &st_file,
        N_THREADS,
        layer_md.mlp.gate_p,
    );
    defer arena.free(buf);
}

// HEADER

pub const HeaderParsingError = error{ FailedRead, MalformedHeader };

fn parseHeader(
    io: Io,
    allocator: std.mem.Allocator,
    st_file: *const Io.File,
) ![16]dequantiZ.metadata.LayerMetaData {

    // header size
    var tmp: [8]u8 = .{ 0, 0, 0, 0, 0, 0, 0, 0 };
    const n = st_file.readPositionalAll(io, &tmp, 0) catch |e| {
        std.log.err("{}", .{e});
        return HeaderParsingError.FailedRead;
    };
    if (n != 8) {
        std.log.err("{s}", .{"couldn't read header size"});
        return HeaderParsingError.MalformedHeader;
    }
    const header_size: u64 = @bitCast(tmp);
    // std.log.info("header size = {}", .{header_size});

    // header content
    var reader_buf: [1024]u8 = undefined;
    var reader = st_file.reader(io, &reader_buf);
    const header_blob = reader.interface.readAllocAll(allocator, @intCast(header_size + 8)) catch |e| {
        std.log.err("{}", .{e});
        return HeaderParsingError.FailedRead;
    };
    if (header_blob.len != @as(usize, header_size + 8)) {
        std.log.err("...", .{});
        return HeaderParsingError.MalformedHeader;
    }

    const md: [16]dequantiZ.metadata.LayerMetaData = try dequantiZ.metadata.buildLayerMetaData(allocator, header_blob);

    return md;
}
