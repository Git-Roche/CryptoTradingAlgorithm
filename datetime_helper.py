
from datetime import datetime, timedelta

def round_to_nearest_hour(dt) -> datetime:
    """
    This takes in a datetime (dt) and round it to the nearest hour
    
    ### Returns

    datetime

    ### Example
    >>> from datetime import datetime
    >>> dt = datetime(2021, 3, 6, 23, 3, 43, 123)
    >>> print(str(dt))

    2021-03-06 23:03:43.000123
    >>> dt = round_to_nearest_hour(dt)
    >>> print(str(dt))

    2021-03-06 23:00:00 
    """
    dt_new = datetime(
        dt.year, 
        dt.month, 
        dt.day, 
        dt.hour, 
        0, 
        0, 
        0
    )

    if dt.minute >= 30:
        #round up
        dt_new += timedelta(hours=1)

    return dt_new