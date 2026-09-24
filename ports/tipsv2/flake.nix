{
  description = "PyTorch reference environment for porting TIPSv2 (CPU only, no CUDA)";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      systems = [ "aarch64-darwin" "x86_64-darwin" "x86_64-linux" "aarch64-linux" ];
      forAllSystems = f: nixpkgs.lib.genAttrs systems (system:
        f (import nixpkgs { inherit system; config.cudaSupport = false; }));
    in {
      devShells = forAllSystems (pkgs:
        let
          # macOS: the official wheel (CPU/MPS only). Linux: the nixpkgs CPU build,
          # since the Linux wheels pull in the CUDA runtime.
          torch = pkgs.python3Packages.torch;
          python = pkgs.python3.withPackages (ps: [ torch ps.numpy ps.safetensors ps.sentencepiece ]);
        in {
          default = pkgs.mkShellNoCC { packages = [ python ]; };
        });
    };
}
