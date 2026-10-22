{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
        };
      in
      {
        devShells.default = pkgs.mkShell {
          nativeBuildInputs = with pkgs; [
            zig_0_17
            zls
            samply
            uv
          ];
          buildInputs = with pkgs; [
            hwloc.dev # Provides hwloc library
          ];
          shellHook = ''
            export LD_LIBRARY_PATH=${pkgs.hwloc.lib}/lib:$LD_LIBRARY_PATH
          '';
        };
      });
}
