import unittest
import pandas as pd
import datetime

class TestPrediction(unittest.TestCase):
    
    def test_check_time_should_be_same(self):
        today_str = datetime.today().strftime('%Y-%m-%d')
        data = pd.date_range(today_str, periods=1, freq="d")
        print(data)


if __name__ == "__main__":
    print("asd")
    unittest.main()