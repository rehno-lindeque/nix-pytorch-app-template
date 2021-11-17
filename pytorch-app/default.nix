{ lib
, buildPythonApplication
, python
, pytorch
, torchvision
, pandas
, setuptools
, wandb
}:


buildPythonApplication {
  pname    = "pytorch-app";
  version = "1.0";

  src = ./.;
  propagatedBuildInputs = [
    pytorch
    setuptools
    torchvision
    wandb
  ];

  installPhase = ''
    mkdir -p $out/share $out/bin
    cp -r src $out/share

    chmod +x $out/share/src/train/main.py
    wrapPythonProgramsIn $out/share/src/train "$out/share/src/train $pythonPath"
    ln -s $out/share/src/train/main.py $out/bin/train

    chmod +x $out/share/src/inference/main.py
    wrapPythonProgramsIn $out/share/src/inference "$out/share/src/inference $pythonPath"
    ln -s $out/share/src/inference/main.py $out/bin/inference
  '';

}

