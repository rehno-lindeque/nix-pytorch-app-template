{ pkgs ? import <nixpkgs> {}
, ...
}:

let
  pythonEnv = pkgs.python.withPackages (pythonPkgs:
    let
      app = (pythonPkgs.callPackage ./. {});
    in
      with pythonPkgs;
      [
        flake8
      ] ++ app.propagatedBuildInputs
  );
in
pkgs.mkShell {
  buildInputs = with pkgs; [
    black
    mypy
    nixpkgs-fmt
    poetry
    pythonEnv
  ];
  shellHook =
    let
      nc = "\\e[0m"; # No Color
      white = "\\e[1;37m";
    in
      ''
        clear -x
        printf "${white}"
        echo "-------------------------------"
        echo "PyTorch development environment"
        echo "-------------------------------"
        printf "${nc}"
        echo

        alias pip="PIP_PREFIX='$(pwd)/_build/pip_packages' TMPDIR='/tmp' \pip"
        export PYTHONPATH="$(pwd)/_build/pip_packages/${pythonEnv.sitePackages}:$PYTHONPATH"
        export PATH="$(pwd)/_build/pip_packages/bin:$PATH"
        unset SOURCE_DATE_EPOCH
    '';
}
