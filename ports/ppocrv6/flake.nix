{
  description = "transformers + OpenCV reference environment for porting PP-OCRv6 (CPU only, no CUDA)";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      systems = [ "aarch64-darwin" "x86_64-darwin" "x86_64-linux" "aarch64-linux" ];
      forAllSystems = f: nixpkgs.lib.genAttrs systems (system:
        f (import nixpkgs { inherit system; config.cudaSupport = false; }));
    in {
      devShells = forAllSystems (pkgs: {
        default = pkgs.mkShellNoCC {
          packages = [
            (pkgs.python3.withPackages (ps: [
              ps.torch ps.torchvision ps.transformers ps.opencv4 ps.safetensors ps.numpy ps.pillow
            ]))
          ];
        };
      });
    };
}
