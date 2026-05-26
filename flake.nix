{
    description = "Flake for the creation of SCNF";

    inputs = {
        nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
        gitpkgs.url = "github:Voivodic/nix-derivations/main";
        gitpkgs.inputs.nixpkgs.follows = "nixpkgs";
    };

    outputs = { self, nixpkgs, gitpkgs, ... } @ inputs:
    let
        # Set the system and the pkgs used
        system = "x86_64-linux";
        pkgs = import nixpkgs {
            inherit system;
            config = {
                allowUnfree = true;
                cudaSupport = true;
            };
            overlays = [ gitpkgs.overlays.default ];
        };

        # Set the python version used
        pyPkgs = pkgs.python314Packages;

        # Build the SCNF package
        scnf =  pyPkgs.buildPythonPackage {
            pname = "SCNF";
            version = "0.1.0";
            format = "pyproject";

            src = ./.;

            nativeBuildInputs = with pyPkgs; [
                setuptools
            ];

            propagatedBuildInputs = with pyPkgs; [
                jaxtyping
                optax
                h5py
                equinox
                e3nn-jax
                diffrax
            ];

            nativeCheckInputs = with pyPkgs; [
                pytest
            ];

            doCheck = true;

            checkPhase = ''
                runHook preCheck
                
                export JAX_PLATFORMS=cpu
                
                pytest tests/
                
                runHook postCheck
            '';

            pythonImportsCheck = [ "scnf" ];

            meta = {
                description = "Steerable continuous normalizing flows.";
                homepage = "https://github.com/Voivodic/scnf";
            };
        };
    in
    {
        # Set scnf as the default package
        packages.${system}.default = scnf;

        # Instructions for the creation of a nix shell with scnf
        devShells.${system}.default = pkgs.mkShell {
            inputsFrom = [ scnf ];

            packages = with pyPkgs; [
                getdist
                ruff
                pytest
            ];
        };

        # Configure the "apps" (test and python with scnf) for the flake
        apps.${system} = {
            test = {
                type = "app";
                program = "${pyPkgs.pytest}/bin/pytest";
            };
            
            default = {
                type = "app";
                program = "${pyPkgs.python}/bin/python";
            };
        };
    };
}
