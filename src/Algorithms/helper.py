def convert_time_to_slot(time, slot_length):
    minutes = time // 60
    mod = minutes % slot_length
    
    if mod != 0:
        if mod < slot_length / 2:
            minutes -= mod
        else:
            minutes += slot_length - mod

    slot = minutes // slot_length
    return slot
