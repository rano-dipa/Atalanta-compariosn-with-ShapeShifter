from codec import Codec
from probability_table import ProbabilityModel

class AtalantaDecoder(Codec):
    def __init__(self, PCNT):
        """
        Initialize the decoder with the given probability count table (PCNT).
        """
        self.HIGH = 0xFFFF  # Max value for 16 bits
        self.LOW = 0x0000  # Min value for 16 bits
        self.PCNT = PCNT   # Symbol & Probability Count Table
        self.value = 0     # Decoded value from input bits
        self.input_bits = []  # Input bitstream (to be provided for decoding)

    def mask_16(self, value):
        """
        Mask the given value to ensure it fits within 16 bits.
        """
        return value & 0xFFFF

    def load_initial_value(self):
        """
        Load the initial value by reading the first 16 bits from the input bitstream.
        """
        self.value = 0
        for _ in range(16):  # Read 16 bits
            if self.input_bits:
                bit = self.input_bits.pop(0)
                self.value = (self.value << 1) | bit
                self.value = self.mask_16(self.value)  # Keep `value` within 16 bits
            else:
                raise ValueError("Insufficient bits in the input stream to load initial value.")

    def get_symbol_from_range(self):
        """
        Identify the symbol corresponding to the current range.
        """
        range_val = self.HIGH - self.LOW + 1
        scaled_value = ((self.value - self.LOW + 1) * 1024 - 1) // range_val

        # Find the symbol whose range includes the scaled value
        for entry in self.PCNT:
            if entry['t_low'] <= scaled_value < entry['t_high']:
                return entry
        raise ValueError(f"Scaled value {scaled_value} does not match any range in PCNT.")

    def decode(self, bitstream):
        """
        Decodes the given bitstream into the original symbols.
        :param bitstream: A list of bits representing the encoded input stream.
        :return: A list of decoded symbols.
        """
        self.input_bits = bitstream  # Initialize the input bitstream
        self.load_initial_value()   # Load the initial value from the first 16 bits

        decoded_symbols = []

        while self.input_bits:
            # Step 1: Get the symbol from the current range
            symbol_entry = self.get_symbol_from_range()
            decoded_symbols.append(symbol_entry['v_min'])

            # Step 2: Update HIGH and LOW based on the symbol's range
            range_val = self.HIGH - self.LOW + 1
            self.HIGH = self.LOW + ((range_val * symbol_entry['t_high']) >> 10) - 1
            self.LOW = self.LOW + ((range_val * symbol_entry['t_low']) >> 10)

            # Ensure HIGH and LOW remain 16-bit values
            self.HIGH = self.mask_16(self.HIGH)
            self.LOW = self.mask_16(self.LOW)

            # Step 3: Adjust HIGH and LOW by processing input bits to stabilize the range
            while True:
                if self.HIGH < 0x8000:  # Case 1: MSB of both HIGH and LOW is 0
                    self.HIGH = (self.HIGH << 1) | 1
                    self.LOW = (self.LOW << 1)
                    self.value = (self.value << 1) | self._consume_bit()

                elif self.LOW >= 0x8000:  # Case 2: MSB of both HIGH and LOW is 1
                    self.HIGH = (self.HIGH << 1) | 1
                    self.LOW = (self.LOW << 1)
                    self.value = (self.value << 1) | self._consume_bit()

                elif self.LOW >= 0x4000 and self.HIGH < 0xC000:  # Case 3: Underflow
                    self.HIGH = ((self.HIGH << 1) & 0xFFFF) | 0x8001
                    self.LOW = ((self.LOW << 1) & 0x7FFF)
                    self.value = ((self.value << 1) | self._consume_bit()) & 0x7FFF

                else:
                    break

                # Mask all values to keep them 16 bits
                self.HIGH = self.mask_16(self.HIGH)
                self.LOW = self.mask_16(self.LOW)
                self.value = self.mask_16(self.value)

        return decoded_symbols

    def _consume_bit(self):
        """
        Consumes the next bit from the input stream. If no bits are left, appends a 0.
        """
        if self.input_bits:
            return self.input_bits.pop(0)
        return 0
