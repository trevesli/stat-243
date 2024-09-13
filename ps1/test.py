def get_all_citations(scholar_id):
    """Constructs an http GET request
    (and submits that request) to get ALL
    scholar's citations from Google Scholar

    Args:
      scholar_id (str): Google scholar ID

    Returns:
      citations_mega (python object): All the scholar's citations
    """

    # We need to increment "cstart" in the search term until the page returns the string "No citations found."
    cstart_inc = 0

    # The while loop was with assistance from ChatGPT
    # Finding "cstart" and "pagesize" keywords was from Stack Overflow
    while True:
        # Define the url
        url = (
            "https://scholar.google.com/citations?user="
            + str(scholar_id)
            + "&hl=en&oi=ao&cstart="
            + str(cstart_inc)
            + "&pagesize=100"  # 100 seems to be max allowable page size
        )

        # Define the saved html; give it a filename
        saved_html = f"scholar_mega_cstart{cstart_inc}.html"

        # Download the page
        subprocess.run(["wget", "-O", saved_html, url], check=True)

        # Check if the file contains any results
        with open(saved_html, "r", encoding="ISO-8859-1") as file:
            content = file.read()

            # # Look for the message that indicates no more results
            if "There are no articles in this profile." in content:
                print("No more results found. Ending download.")

                # Delete the last downloaded file
                subprocess.run(["rm", saved_html])

                break

        cstart_inc += 100

    # Combine all downloaded webpages into 1
    subprocess.run("cat scholar_mega_cstart* > scholar_mega_combined.html", shell=True)

    # Delete all the intermediate htmls
    subprocess.run("rm scholar_mega_cstart*", shell=True)

    return 


# We can then use this code with the code from Question 4c
# e.g. get_scholar_info("scholar_mega_combined.html'")
