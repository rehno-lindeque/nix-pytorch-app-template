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
        pip
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
        echo "USAGE:"
        echo "  pip install .         # get started developing"
        echo "  python app/train      # run training for local development"
        echo "  python app/inference  # run inference for local development"
        echo
        echo "To do a full training run, build the nix model"
        echo

        alias pip="PIP_PREFIX='$(pwd)/_build/pip_packages' TMPDIR='/tmp' \pip"
        export PYTHONPATH="$(pwd)/_build/pip_packages/${pythonEnv.sitePackages}:$PYTHONPATH"
        export PATH="$(pwd)/_build/pip_packages/bin:$PATH"
        unset SOURCE_DATE_EPOCH
    '';
}
