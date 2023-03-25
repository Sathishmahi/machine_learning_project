from collections import namedtuple

DataInjectionArtifacts=namedtuple("DataInjectionArtifacts", ["train_file_path",
                                "test_file_path","is_injected","message"])