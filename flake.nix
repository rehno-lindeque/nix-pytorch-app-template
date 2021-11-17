{
  description = "Nix templates for creating various simple pytorch apps";
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-21.05-small";
    utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, utils, ... }:
    let
      eachDefaultEnvironment = f: utils.lib.eachDefaultSystem
        (
          system:
          f {
            inherit system;
            pkgs = import nixpkgs { inherit system; };
          }
        );
    in
    eachDefaultEnvironment ({ pkgs, system }: {

      devShell = import ./shell.nix { inherit pkgs; };

    }) // {

      templates = {
        pytorch-app = {
          path = ./pytorch-app;
          description = "A template for a simple pytorch app";
        };

      };

      defaultTemplate = self.templates.pytorch-app;

    };
}
