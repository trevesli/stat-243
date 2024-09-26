filename = sm.distributions.empirical_distribution.__file__

with open(filename, "r") as f:
    for line_num, line in enumerate(f, start=1):
        if "monotone_fn_inverter" in line:
            print(f"Line {line_num}: {line.strip()}")
