//! By convention, root.zig is the root source file when making a package.
const std = @import("std");

pub const decode = @import("decode.zig");
pub const metadata = @import("metadata.zig");
pub const tensor = @import("tensor.zig");

