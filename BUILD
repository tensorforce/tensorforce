package(default_visibility = ["//visibility:public"])

py_library(
    name = "tforce",
    imports = [":tensorforce"],
    srcs = glob(["tensorforce/**/*.py"])
)

py_binary(
    name = "lab_runner",
    srcs = ["tensorforce/examples/deepmind_lab.py"],
    data = ["//:deepmind_lab.so"],
    main = "tensorforce/examples/deepmind_lab.py",
    deps = [":tforce"]
)



