package(default_visibility = ["//visibility:public"])

env_args = [
  "--agent-config examples/configs/dqn_network.json",
  "--network-config examples/configs/dqn_network.json"
]

py_library(
    name = "tensorforce",
    imports = [":tensorforce"],
    data = ["//tensorforce:examples/configs/dqn_agent.json",
    "//tensorforce:examples/configs/dqn_network.json"],
    srcs = glob(["tensorforce/**/*.py"])
)

py_binary(
    name = "lab_runner",
    srcs = ["examples/lab_main.py"],
    data = ["//:deepmind_lab.so"],
    main = "examples/lab_main.py",
    deps = [":tensorforce"]
)



