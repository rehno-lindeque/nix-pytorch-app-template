{
  # Generated via github:rehno-lindeque/nix-pytorch-app-template
  description = "Simple PyTorch application";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/46251a79f752ae1d46ef733e8e9760b6d3429da4";
    utils.url = "github:numtide/flake-utils";
    mlPkgsSrc.url = "github:rehno-lindeque/ml-pkgs";
  };

  outputs = { self, nixpkgs, utils, mlPkgs, ... }:

    let
      inherit (nixpkgs) lib;

      eachDefaultEnvironment = f: utils.lib.eachDefaultSystem
        (
          system:
          f {
            inherit system;
            pkgs = (import nixpkgs { inherit system; config.allowUnfree = true; }).extend self.overlay;
          }
        );

    in
    eachDefaultEnvironment ({ pkgs, system }: {

      devShell = import ./shell.nix {
        inherit pkgs lib;
        inherit (self.packages."${system}") pytorchApp;
      };

      packages.pytorchApp = pkgs.callPackage ./. {};

      defaultPackage = (self.packages."${system}").pytorch-template-offset;

    }) // {

      overlay = final: prev: {
        python = final.python38.override {
          packageOverrides =
            lib.composeExtensions
              (_: _: { inherit (final.linuxPackages) nvidia_x11; })
              mlPkgs.overlay;
        };
      };
    };
}
