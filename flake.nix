{
    description = "Flake for the creation of a shell with SCNF";

    inputs = {
        nixpkgs.url = github:NixOS/nixpkgs/nixos-24.11;
    };

    outputs = { self, nixpkgs, ... } @ inputs: 
    let
        # Set the system and the pkgs used
        system = "x86_64-linux";
        pkgs = import nixpkgs { inherit system; };

        # Install e3nn_jax 
        e3nn = pkgs.python312Packages.buildPythonPackage {
            pname = "e3nn_jax";
            version = "0.20.7";
            format = "pyproject";
            src = pkgs.fetchurl {
                url = "https://github.com/e3nn/e3nn-jax/archive/refs/heads/main.zip";
                sha256 = "sha256-xLubAj5EHKNdPUm/QhzYP9nFm+V2lRS2FEBtaZTH1s4";
            };
            propagatedBuildInputs = [
                pkgs.python312Packages.numpy
                pkgs.python312Packages.jax
                pkgs.python312Packages.jaxlib
                pkgs.python312Packages.setuptools
                pkgs.python312Packages.setuptools_scm
                pkgs.python312Packages.attrs
                pkgs.python312Packages.sympy
            ];
        };

        # Install diffrax
        diffrax = pkgs.python312Packages.buildPythonPackage {
            pname = "diffrax";
            version = "0.5.0";
            format = "pyproject";
            src = pkgs.fetchPypi {
                pname = "diffrax";
                version = "0.5.0";
                sha256 = "sha256-LmZwG1RXmIGK80qRMoh7NUyTtbR+EzyZhuI6fPYAqTc";
            };
            propagatedBuildInputs = [
                pkgs.python312Packages.equinox
                pkgs.python312Packages.lineax
                pkgs.python312Packages.optimistix
                pkgs.python312Packages.hatchling
            ];
        };

        # Install SCNF
        scnf =  pkgs.python312Packages.buildPythonPackage {
            pname = "SCNF";
            version = "0.1.0";
            format = "pyproject";
            src = ./.;
            propagatedBuildInputs = [
                pkgs.python312Packages.typing
                pkgs.python312Packages.tqdm
                pkgs.python312Packages.jaxtyping
                pkgs.python312Packages.optax
                e3nn
                diffrax
            ];
            meta = {
                description = "Steerable continuous normalizing flows.";
                homepage = "https://github.com/Voivodic/SCNF";
            };
        };
    in
    {
        # Instructions for the creation of the shell
        devShells.${system}.default = pkgs.mkShell{
            buildInputs = [
                pkgs.python312
                scnf
            ];

            shellHook = ''
            '';
        };
    };
}
