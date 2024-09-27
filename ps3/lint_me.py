class Chunk:
    def __init__(self, transcript):
        self.transcript = transcript

    def get_debate_year(self):
        """
        Gets the debate year.

        Returns:
            int: The year of the debate
        """
        year_pattern = r"^.+?(\d{4})"
        year_matches = re.findall(year_pattern, self.transcript)
        matches_list = [int(match) for match in year_matches]

        return matches_list[0]

    def strip_preamble(self):
        """
        Strips the transcript preamble up to the first time a name in `moderator`
        is encountered.

        Returns:
            str: Transcript with preamble removed.
        """
        preamble_pattern = (
            r"^(.*?)(?:"
            + "|".join(re.escape(moderator) for moderator in moderators)
            + r"):\s*(.*)"
        )
        preamble_match = re.search(preamble_pattern, self.transcript)
        self.transcript = self.transcript[preamble_match.end(1) :].strip()
        return self.transcript

    def get_chunks(self):
        """
        Isolates and consolidates chunks of dialogue from the transcript
        using regex.

        Returns:
            list: A list of lists, where inner list contains the speaker's
                  identifier and their consolidated speech chunk.
        """
        # Use regex to find all chunks
        chunk_pattern = r"([A-Z]+): (?<=[A-Z]: )(.*?)(?=[A-Z]*:|$)"
        chunk_matches = re.findall(chunk_pattern, self.transcript)
        matches_list = [list(match) for match in chunk_matches]

        # Consolidate chunks if spoken in a row by the same speaker
        i = 0
        while i < len(matches_list) - 1:
            # Check if the speaker names (the first element) are the same
            if matches_list[i][0] == matches_list[i + 1][0]:
                matches_list[i][1] += " " + matches_list[i + 1][1]
                del matches_list[i + 1]
            else:
                i += 1
        """for i in range(len(matches_list) - 1):
      if matches_list[i][0] == matches_list[i+1][0]:
        matches_list[i][1] += matches_list[i+1][1]
        matches_list[i+1][0] = None
        matches_list[i+1][1] = None
    
    # Drop rows where the values are "None"
    filtered_chunks = [chunk for chunk in matches_list 
                      if chunk[0] is not None 
                      and chunk[1] is not None]"""

        return matches_list

    def count_chunks(self, chunks):
        """
        Counts the number of chunks per speaker for each debate.
        Only speakers in the `speakers` list are considered.
        Also drop keys where value is less than 5.

        Returns:
            dict: A dict where the keys are speaker names, and the values
                  are the counts of chunks per speaker.
        """
        speaker_count = {}

        for chunk in chunks:
            speaker = chunk[0]  # The first element is the speaker

            # Ensure speaker is only counted if they are in `speakers` list
            if speaker in speaker_count and speaker in speakers:
                # Increment counter for a particular speaker
                speaker_count[speaker] += 1
            elif speaker in speakers:
                # Initialise value for first time speaker is encoutered
                speaker_count[speaker] = 1

        # Remove items in the dict with values less than 5,
        # since these likely aren't real chunks
        speaker_count = {
            key: value for key, value in speaker_count.items() if value >= 5
        }

        return speaker_count

    def check_chunk_repeats(self, debate_chunked):
        """
        As a sanity check, counts the occurrences of the speaker in
        consecutive chunks being the same.

        Returns:
            int: Count where speakers in consecutive chunks is the same.
        """
        # See if chunk and chunk+1 speakers are the same
        repeat_counts = 0
        for i in range(len(debate_chunked) - 1):
            if debate_chunked[i][0] == debate_chunked[i + 1][0]:
                # Print out the offending lines
                print(f"{debate_chunked[i][0]}: {debate_chunked[i][1]}")
                print(f"{debate_chunked[i+1][0]}: {debate_chunked[i+1][1]}")
                repeat_counts += 1
        return repeat_counts


2000: {'MODERATOR': 60, 'GORE': 49, 'BUSH': 56}
2004: {'SCHIEFFER': 58, 'KERRY': 31, 'BUSH': 29}
2008: {'SCHIEFFER': 57, 'MCCAIN': 65, 'OBAMA': 52}
2012: {'LEHRER': 82, 'OBAMA': 56, 'ROMNEY': 71}
2016: {'HOLT': 97, 'CLINTON': 87, 'TRUMP': 124}
2020: {'WALLACE': 246, 'BIDEN': 269, 'TRUMP': 341}
    
2000: {'MODERATOR': 60, 'GORE': 49, 'BUSH': 56}
2004: {'SCHIEFFER': 57, 'KERRY': 31, 'BUSH': 29}
2008: {'SCHIEFFER': 55, 'MCCAIN': 60, 'OBAMA': 45}
2012: {'LEHRER': 78, 'OBAMA': 42, 'ROMNEY': 57}
2016: {'HOLT': 97, 'CLINTON': 87, 'TRUMP': 123}
2020: {'WALLACE': 245, 'BIDEN': 267, 'TRUMP': 338}
    
2000: {'MODERATOR': 60, 'GORE': 49, 'BUSH': 56}
2004: {'SCHIEFFER': 57, 'KERRY': 31, 'BUSH': 29}
2008: {'SCHIEFFER': 55, 'MCCAIN': 60, 'OBAMA': 45}
2012: {'LEHRER': 76, 'OBAMA': 42, 'ROMNEY': 54}
2016: {'HOLT': 97, 'CLINTON': 87, 'TRUMP': 123}
2020: {'WALLACE': 245, 'BIDEN': 266, 'TRUMP': 337}