In the code below, I first find and strip out non-spoken text, which is normally indicated by single words being in parentheses or square brackets. But I ignore words that are two letters or fewer, as it seems to flag party affiliation (like "(D)" or "(R)") or state names (like "(MA)" or "(AZ)").

```{python}
#| cache: true
def strip_non_spoken(debates_all):
  """
  Strips non-spoken text (e.g. "laughter", "applause") from a list of debate 
  transcripts, using regex. Returns a cleaned-up version of debate texts

  Parameters:
  ----------
  debates_all : list of str
      A list of strings, where each string is a transcript of a debate.

  Returns:
  -------
  list of str
      A list of strings where the non-spoken text has been removed.

  Example:
  -------
  >>> debates_body = [
          "They will applaud as we welcome the two candidates, 
          Governor Bush and Vice President Gore.(Applause)",
          "Well, just listen to what you heard. (laughter)"
      ]
  >>> strip_non_spoken(debates)
  ["They will applaud as we welcome the two candidates, 
  Governor Bush and Vice President Gore.",
  "Well, just listen to what you heard."]
  """

  import re

  # Find all non-spoken text similar to "Laughter" or "Applause"
  # Do this by regex-ing all single words that appear within parentheses 
  # and square brackets
  # Exclude words with two or fewer characters
  non_spoken_pattern = r"[\(\[]\w{3,}[\)\]]"
  non_spoken_matches = [re.findall(non_spoken_pattern, debate) 
                        for debate in debates_all]

  # Flatten the list of lists using list comprehension
  non_spoken_matches = [match for debate in non_spoken_matches 
                        for match in debate]

  # Create new regex pattern based on non_spoken_matches
  non_spoken_matches_pattern = '|'.join(map(re.escape, non_spoken_matches))

  # Strip out all non-spoken text using regex
  # Strip out leading and trailing whitespaces
  debates_body_stripped = [re.sub(non_spoken_matches_pattern, '', debate).strip() 
                        for debate in debates_all]

  return debates_body_stripped

# Apply function to our list of transcripts
debates_body_stripped = strip_non_spoken(debates_body)

# Sanity check: compare occurrences of "applause" and "laughter"
applause_counts_orig = [debate.lower().count("applause") 
                        for debate in debates_body]
applause_counts_stripped = [debate.lower().count("applause") 
                        for debate in debates_body_stripped]
laughter_counts_orig = [debate.lower().count("laughter") 
                        for debate in debates_body]
laughter_counts_stripped = [debate.lower().count("laughter") 
                        for debate in debates_body_stripped]

print("Occurrences of \"applause\" in original transcript:", 
      applause_counts_orig)
print("Occurrences of \"applause\" in stripped transcript:", 
      applause_counts_stripped)
print("Occurrences of \"laughter\" in original transcript:", 
      laughter_counts_orig)
print("Occurrences of \"laughter\" in stripped transcript:", 
      laughter_counts_stripped)

I then subdivide each transcript based on the idea that each chunk will be marked by a speaker (candidate or moderator; stylised in all caps) followed by a colon. The list of `candidates` is in the provided `ps3prob3.py` file, as is the name of each of the moderators. 

```{python}
#| cache: true

# Get the names of debaters for years of interest 
candidates_names = [names for debater in candidates for names in debater.values()]

# Retrieve only unique names, for both candidates and moderators
candidates_names = list(set(candidates_names))
moderator_names = list(set(moderators))

# Combine to make a list of speakers
speakers = candidates_names + moderator_names

print(candidates_names)
print(moderator_names)
```

To retrieve the chunks, I break up each debate transcript by isolating a speaker and their response in my `get_chunks()` function, using regex.

```{python}
def get_chunks(string):

  # Use regex to find all chunks
  pattern = r"([A-Z]+): (?<=[A-Z]: )(.*?)(?=[A-Z]*:|$)"
  matches = re.findall(pattern, string)
  matches_list = [list(match) for match in matches]

  # Consolidate chunks if spoken in a row by the same speaker
  for i in range(len(matches_list) - 1):
    if matches_list[i][0] == matches_list[i+1][0]:
      matches_list[i][1] += matches_list[i+1][1]
      matches_list[i+1][0] = None
      matches_list[i+1][1] = None
  
  # Drop rows where the values are "None"
  filtered_chunks = [chunk for chunk in matches_list 
                    if chunk[0] is not None 
                    and chunk[1] is not None] 
  
  return filtered_chunks 

# Create a list that allocates a variable to each debate
debates_list = []

# Get chunks for each debate
for i, debate in zip(debates_body_stripped, debates_list):
  ... #TODO  

get_chunks(debates_body_stripped[0])

# Do some sanity checks
```

I then create a function `count_chunks()` that counts how many chunks can be attributed to each of the three speakers in a debate:

```{python}
#| cache: true
def count_chunks(list):

  for chunk in list:
    ...#TODO

```

I define a class called Chunks to store/structure the spoken chunks:

```{python}
class Chunk:
  def __init__(self, speaker, text):
    self.speaker = speaker
    self.text = text
```






class Chunk:
    def __init__(self, speaker, text):
        self.speaker = speaker
        self.text = text

    def __str__(self):
        return f"{self.speaker}: {self.text}"

class Debate:
    def __init__(self, year, transcript):
        self.year = year
        self.transcript = transcript
        self.chunks = []

    def process_transcript(self):
        # Process the transcript to create chunks
        # Strip out non-spoken text and combine chunks
        pass

    def get_chunk_counts(self):
        # Return a dictionary with speaker names and their chunk counts
        chunk_counts = {}
        for chunk in self.chunks:
            if chunk.speaker in chunk_counts:
                chunk_counts[chunk.speaker] += 1
            else:
                chunk_counts[chunk.speaker] = 1
        return chunk_counts

# Example usage:
debates = [
    Debate(2000, "transcript_text_2000"),
    Debate(2004, "transcript_text_2004"),
    # Add more debate instances
]

for debate in debates:
    debate.process_transcript()
    counts = debate.get_chunk_counts()
    print(f"Chunk counts for debate {debate.year}: {counts}")

