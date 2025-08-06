from wetterdienst.provider.dwd.observation import DwdObservationMetadata

# print resolutions, datasets and parameters
for resolution in DwdObservationMetadata:
    print(f"{2 * "\t"}{resolution.name}", "\n")
    for dataset in resolution:
        print(f"{3 * "\t"}{dataset.name}", "\n")
        for parameter in dataset:
            print(f"{4 * "\t"}{parameter.name}", "\n")
