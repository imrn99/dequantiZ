const std = @import("std");
const Io = std.Io;

const dequantiZ = @import("dequantiZ");

const TENSOR_FILE = "model.safetensors";

pub fn main(init: std.process.Init) !void {
    var layer_id: u4 = 0;
    var n_threads: usize = 1;
    var args = init.minimal.args.iterate();
    _ = args.next(); // discard "./zig-out/bin/dequantiz"

    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "--n-threads")) {
            const raw = args.next() orelse {
                std.log.err("Unspecified thread number", .{});
                return;
            };
            n_threads = std.fmt.parseInt(usize, raw, 10) catch {
                std.log.err("Could not parse thread number: {s}", .{raw});
                return;
            };
        } else if (std.mem.eql(u8, arg, "--layer-id")) {
            const raw = args.next() orelse {
                std.log.err("Unspecified layer ID (expected usize in base 10)", .{});
                return;
            };
            layer_id = std.fmt.parseInt(u4, raw, 10) catch {
                std.log.err("Could not parse layer ID (expected u4 in base 10): {s}", .{raw});
                return;
            };
        } else {
            std.log.err("Unknown argument: {s}", .{arg});
            return;
        }
    }

    const arena: std.mem.Allocator = init.arena.allocator();
    const io = init.io;
    const st_file = Io.Dir.cwd().openFile(io, TENSOR_FILE, .{}) catch |e|
        {
            std.log.err("while opening tensor file: {}", .{e});
            return;
        };
    defer st_file.close(io);

    const md = try parseHeader(io, arena, &st_file);
    const layer_md = md[@intCast(layer_id)];

    const buf = try dequantiZ.tensor.buildTensorData(
        io,
        arena,
        &st_file,
        n_threads,
        layer_md.mlp.gate_p,
    );
    defer arena.free(buf);

    // for (0..16) |i| {
    //     std.log.info("buf[{}]: 0x{x}", .{i, buf[i].toU16()});
    // }
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
