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
        gpkgs = {
            python314 = pkgs.python314Packages;
        };

        # Install SCNF
        scnf =  pkgs.python314Packages.buildPythonPackage {
            pname = "SCNF";
            version = "0.1.0";
            format = "pyproject";

            src = ./.;

            buildInputs = [
                pkgs.python314Packages.setuptools
            ];

            nativeCheckInputs = [
                pkgs.python314Packages.pytest
            ];

            propagatedBuildInputs = [
                pkgs.python314
                pkgs.python314Packages.jaxtyping
                pkgs.python314Packages.optax
                pkgs.python314Packages.h5py
                pkgs.python314Packages.equinox
                gpkgs.python314.e3nn-jax
                gpkgs.python314.diffrax
            ];

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
        devShells.${system}.default = pkgs.mkShell{
            buildInputs = [
                gpkgs.python314.getdist
                scnf
            ];
        };
    };
}
