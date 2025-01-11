from codec import Codec
from probability_table import ProbabilityModel

class AtalantaEncoder(Codec):
    """
    A class representing the Atalanta Encoder, responsible for encoding a stream of data using an arithmetic coding technique.
    
    Attributes:
        UBC (int): The underflow bit counter.
        OFS_out (list): The list storing the offset bit stream.
        OFS_r (list): The list storing the offset bit length stream.
        CODE_out (list): The list storing the symbol stream.
        CODE_c (list): The list storing the symbol length stream.
        PCNT (list): The model representing probability values for each character.
    """

    def __init__(self, model):
        """
        Initializes the AtalantaEncoder with the given model.

        Args:
            model (list): The probability model to use for encoding.
        """
        self.UBC = 0  # Underflow bit counter initialized to 0.

        self.OFS_out = []  # List for storing the offset bit stream.
        self.OFS_r = []  # List for storing the offset bit length stream.

        self.CODE_out = []  # List for storing the symbol stream.
        self.CODE_c = []  # List for storing the symbol length stream.

        self.PCNT = ProbabilityModel(model)  # Probability model used for encoding.

    def output_bit(self, bit):
        """
        Outputs a single bit to the symbol stream.

        Args:
            bit (int): The bit (0 or 1) to output.
        """
        self.CODE_out.append(bit)

    def output_bit_plus_pending(self, bit):
        """
        Outputs a bit along with any pending underflow bits.

        Args:
            bit (int): The bit (0 or 1) to output.

        Returns:
            int: The updated underflow bit counter.
        """
        self.output_bit(bit)  # Output the MSB.
        while self.UBC > 0:
            self.output_bit(1 - bit)  # Output the inverse of the current bit for all pending bits.
            self.UBC -= 1
        return self.UBC + 1

    def encode(self, input_stream):
        """
        Encodes an input stream of symbols using arithmetic encoding.

        Args:
            input_stream (iterable): An iterable containing the stream of symbols to encode.

        Raises:
            ValueError: If a character in the input stream is not found in the probability model.
        """
        for c in input_stream:
            # Step 1: Get the probability entry for the current symbol.
            PCNT_row = self.PCNT.get_probability_of_symbol(c)
            if PCNT_row is None:
                raise ValueError(f"Character {c} not found in the probability model.")

            # Step 2: Calculate the offset for the symbol and check its validity.
            offset = c - PCNT_row['v_min']
            if offset.bit_length() > PCNT_row['OL']:
                raise ValueError(f"Offset {offset} is larger than OL.")
            else:
                # Append the offset and its length to the corresponding streams.
                self.OFS_out.append(offset)
                self.OFS_r.append(PCNT_row['OL'])

            # Step 3: Update HIGH and LOW bounds based on the symbol's probability range.
            range_val = self.HIGH - self.LOW + 1
            self.HIGH = self.LOW + ((range_val * PCNT_row['t_high']) >> 10) - 1
            self.LOW = self.LOW + ((range_val * PCNT_row['t_low']) >> 10)

            # Step 4: Perform arithmetic encoding by shifting HIGH and LOW.
            while True:
                if self.HIGH < 0x8000:  # Case 1: MSB of both HIGH and LOW is 0.
                    self.output_bit_plus_pending(0)
                    self.LOW <<= 1
                    self.LOW = self.mask_16(self.LOW)
                    self.HIGH <<= 1
                    self.HIGH = self.mask_16(self.HIGH)
                    self.HIGH |= 1  # Set LSB of HIGH to 1.
                elif self.LOW >= 0x8000:  # Case 2: MSB of both HIGH and LOW is 1.
                    self.output_bit_plus_pending(1)
                    self.LOW <<= 1
                    self.LOW = self.mask_16(self.LOW)
                    self.HIGH <<= 1
                    self.HIGH = self.mask_16(self.HIGH)
                    self.HIGH |= 1  # Set LSB of HIGH to 1.
                elif self.LOW >= 0x4000 and self.HIGH < 0xC000:  # Case 3: Handling overlapping MSBs.
                    self.UBC += 1  # Increment the underflow bit counter.
                    self.LOW <<= 1
                    self.LOW &= 0x7FFF  # Set MSB of LOW to 0.
                    self.LOW = self.mask_16(self.LOW)

                    self.HIGH <<= 1
                    self.HIGH = self.mask_16(self.HIGH)
                    self.HIGH |= 0x8001  # Set MSB and LSB of HIGH to 1.
                else:
                    # If no matching condition, break out of the loop.
                    break

        # Step 5: Finalize the encoding process for the last symbol.
        self.UBC += 1
        if self.LOW < 0x4000:
            self.output_bit_plus_pending(0)
        else:
            self.output_bit_plus_pending(1)

    def finalize(self):
        """
        Finalizes the encoding process and returns the resulting streams.

        Returns:
            tuple: A tuple containing the symbol stream (CODE_out), offset stream (OFS_out), and offset length stream (OFS_r).
        """
        return self.CODE_out, self.OFS_out, self.OFS_r
