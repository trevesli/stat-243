exprs = [
    r"I",
    r"we",
    r"America(n)?",
    r"democra(cy|tic)?",
    r"republic",
    r"Democrat(ic)?",
    r"Republican",
    r"free(dom)?",
    r"terror(ism)?",
    r"safe(r|st|ty)?",
    r"(Jesus|Christ|Christian)",
]


# Create a new list to aggregate each candidates words
expr_list = []
[expr_list.append([name, []]) for name in candidates_unique]

for name in expr_list:
    for speakers in extracted_words.values():
        # Loop through each speaker in each debate year
        for speaker in speakers:
            if speaker[0] == name[0]:
                name[1].extend(speaker[1])
                break

# Create a new list to store counts of each expression for each candidate
expr_counts = []
for i in range(len(expr_list)):
    inner = [expr_list[i][0], []]
    expr_counts.append(inner)

# Now count the occurrences that `exprs` appear for each candidate
for i, name in enumerate(expr_counts):
    for expr in exprs:
        count = 0

        # Replace curly brackets with square brackets
        pattern = expr
        matches = re.findall(pattern, " ".join(expr_list[i][1]))

        # Count the occurrences of the word
        count += len(matches)

        expr_counts[i][1].append(count)


# Initialise dataframe
df = pd.DataFrame(columns=["Candidate"] + exprs)

# Append the counts of `exprs` to the dataframe
for candidates in expr_counts:
    # Create new row with candidate name and their respective counts
    new_row = [candidates[0]] + candidates[1]

    # Append the new row to the DataFrame
    df.loc[len(df)] = new_row

display(df)
