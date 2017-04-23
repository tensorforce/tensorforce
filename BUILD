package(default_visibility = ["//visibility:public"])

py_library(
    name = "tensorforce",
    imports = [":tensorforce"],
    data = ["//tensorforce:tensorforce/examples/configs/dqn_agent.json",
    "//tensorforce:tensorforce/examples/configs/dqn_network.json"],
    srcs = glob(["tensorforce/**/*.py"])
)

py_binary(
    name = "lab_runner",
    srcs = ["tensorforce/examples/lab_main.py"],
    data = ["//:deepmind_lab.so"],
    main = "tensorforce/examples/lab_main.py",
    deps = [":tensorforce"]
)



