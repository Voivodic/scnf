{
    description = "A very basic flake";

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
            src = nixpkgs.fetchPypi {
                inherit pname version;
            }
        };

        # Install SCNF
        scnf =  pkgs.python312Packages.buildPythonPackage {
            pname = "SCNF";
            version = "0.1.0";
            src = ./.;
            propagatedBuildInputs = [
                pkgs.python312Packages.jax
                pkgs.python312Packages.equinox
                pkgs.python312Packages.jaxtyping
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
