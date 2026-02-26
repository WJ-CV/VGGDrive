
def get_nav_info(angle, ka):
    if abs(ka) < 0.16 or abs(angle) < 2:
        return 'None'
    if abs(angle) >= 2 and abs(angle) < 3.5:
        if angle > 0:
            return 'Front Left'
        else:
            return 'Front Right'

    if angle > 0:
        return 'Turn Left'
    else:
        return 'Turn Right'
    