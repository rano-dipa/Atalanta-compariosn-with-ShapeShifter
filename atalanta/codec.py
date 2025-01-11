class Codec:
    """
    A base class for encoding streams using arithmetic coding.
    This class defines the basic structure for encoding.

    Attributes:
        HIGH (int): The upper bound for the encoding range.
        LOW (int): The lower bound for the encoding range.
        range_length (int): The length of the encoding range (fixed to 16).

    """
    
    def __init__(self, model):
        """
        Initializes the base Encoder class with required parameters for encoding.
        """
        self.HIGH = 0xFFFF  # Upper bound of the encoding range.
        self.LOW = 0x0000  # Lower bound of the encoding range.
        self.UBC = 0  # Underflow bit counter.

    def mask_16(self, value):
        """
        Masks the given value to fit within 16 bits.
        
        Args:
            value (int): The value to mask.
        
        Returns:
            int: The masked value.
        """
        return value & 0xFFFF

    def print_bin_hex_dec(self, string, value):
        """
        Prints a value in binary, hexadecimal, and decimal formats.

        Args:
            string (str): A label to print before the value.
            value (int): The value to print.
        """
        print(f"{string}: ", self.decimal_to_bits(value), 
              self.decimal_to_hex(value), value)

    def decimal_to_bits(self, n, bit_length=None):
        """
        Converts a decimal number to a list of bits.

        Args:
            n (int): The number to convert to bits.
            bit_length (int, optional): If specified, the result will be padded to this bit length.

        Returns:
            list: A list of bits representing the decimal number.

        Raises:
            ValueError: If a negative number is provided or if the bit length is too small.
        """
        if n < 0:
            raise ValueError("Only non-negative integers are supported.")

        bits = bin(n)[2:]  # Convert integer to binary, stripping the '0b' prefix.

        if bit_length is not None:
            if len(bits) > bit_length:
                raise ValueError("Bit length is too small to represent the number.")
            bits = [0] * (bit_length - len(bits)) + list(bits)

        return bits

    def decimal_to_hex(self, decimal_num):
        """
        Converts a decimal number to its hexadecimal string representation.

        Args:
            decimal_num (int): The decimal number to convert.

        Returns:
            str: The hexadecimal string (uppercase, without '0x' prefix).

        Raises:
            ValueError: If the input is not an integer.
        """
        if not isinstance(decimal_num, int):
            raise ValueError("Input must be an integer.")

        return hex(decimal_num).upper()

    def finalize(self):
        """
        Finalizes the encoding process and returns the resulting streams.
        This is a placeholder function for subclasses to implement.

        Returns:
            tuple: A tuple containing the symbol stream (CODE_out).
        """
        raise NotImplementedError("Subclasses must implement the finalize method.")
