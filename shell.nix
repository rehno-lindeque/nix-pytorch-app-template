{ pkgs ? import <nixpkgs> {}
}:

let
  pytorch-app-template = pkgs.callPackage ./. {
    name = "pytorch-app-template";
  };

in
pkgs.mkShell {
  buildInputs = with pkgs; [
    nix
    nixpkgs-fmt
    pytorch-app-template
  ];
  shellHook =
    let
      nc = "\\e[0m"; # No Color
      white = "\\e[1;37m";
    in
     ''
        clear -x
        printf "${white}"
        echo "--------------------------------"
        echo "Template development environment"
        echo "--------------------------------"
        printf "${nc}"
        echo
        ${pytorch-app-template}/bin/pytorch-app-template-help

    '';
}
