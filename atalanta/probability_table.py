class ProbabilityModel:
    """
    A class representing the probability model used for encoding in Atalanta.
    
    Attributes:
        PCNT (list): The probability model containing entries for the symbols.
    """
    
    def __init__(self, model):
        """
        Initializes the ProbabilityModel with a given model.

        Args:
            model (list): A list of probability entries used for encoding.
        """
        self.PCNT = model

    def get_probability_of_symbol(self, c):
        """
        Retrieves the probability model entry for the given character `c`.

        Args:
            c (int): The character whose probability model entry to retrieve.

        Returns:
            dict: A dictionary containing the probability model entry, or None if not found.
        """
        try:
            # Search for the probability range corresponding to the character.
            i = next(i for i, entry in enumerate(self.PCNT) if entry['v_min'] <= c <= entry['v_max'])
            return self.PCNT[i]
        except StopIteration:
            # If no match is found, return None.
            return None
        
    def get_symbol_from_probability_range(self, value, high, low):
        """
        Retrieves the probability model entry based on the current encoding range (LOW to HIGH).
        This method calculates the scaled value corresponding to the current position in the range,
        and finds the symbol whose probability range includes that scaled value.

        Returns:
            dict: A dictionary containing the probability model entry for the symbol, 
                or raises a ValueError if no match is found in the probability model.

        Raises:
            ValueError: If the scaled value does not match any range in the probability model.
        """
        # Calculate the scaled value based on the current LOW and HIGH.
        range_val = high - low + 1  
        scaled_value = ((value - low + 1) * 1024 - 1) // range_val

        # Find the symbol whose range includes the scaled value
        for entry in self.PCNT:
            if entry['t_low'] <= scaled_value < entry['t_high']:
                return entry
        
        # Raise an error if no match is found.
        raise ValueError(f"Scaled value {scaled_value} does not match any range in PCNT.")


