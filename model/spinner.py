# --*-- coding: utf-8 --*--
""" A command line spinner tool. Has 29 different styles of spinners.
"""

class Spinner(object):
    """ A spinner that keeps its own progress.
    """
    spinners = [
        "|/-\\",
        "⠂-–—–-",
        "◐◓◑◒",
        "◴◷◶◵",
        "◰◳◲◱",
        "▖▘▝▗",
        "■□▪▫",
        "▌▀▐▄",
        "▉▊▋▌▍▎▏▎▍▌▋▊▉",
        "▁▃▄▅▆▇█▇▆▅▄▃",
        "←↖↑↗→↘↓↙",
        "┤┘┴└├┌┬┐",
        "◢◣◤◥",
        ".oO°Oo.",
        ".oO@*",
        "🌍🌎🌏",
        "◡◡ ⊙⊙ ◠◠",
        "☱☲☴",
        "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏",
        "⠋⠙⠚⠞⠖⠦⠴⠲⠳⠓",
        "⠄⠆⠇⠋⠙⠸⠰⠠⠰⠸⠙⠋⠇⠆",
        "⠋⠙⠚⠒⠂⠂⠒⠲⠴⠦⠖⠒⠐⠐⠒⠓⠋",
        "⠁⠉⠙⠚⠒⠂⠂⠒⠲⠴⠤⠄⠄⠤⠴⠲⠒⠂⠂⠒⠚⠙⠉⠁",
        "⠈⠉⠋⠓⠒⠐⠐⠒⠖⠦⠤⠠⠠⠤⠦⠖⠒⠐⠐⠒⠓⠋⠉⠈",
        "⠁⠁⠉⠙⠚⠒⠂⠂⠒⠲⠴⠤⠄⠄⠤⠠⠠⠤⠦⠖⠒⠐⠐⠒⠓⠋⠉⠈⠈",
        "⢄⢂⢁⡁⡈⡐⡠",
        "⢹⢺⢼⣸⣇⡧⡗⡏",
        "⣾⣽⣻⢿⡿⣟⣯⣷",
        "⠁⠂⠄⡀⢀⠠⠐⠈",
        "🌑🌒🌓🌔🌕🌝🌖🌗🌘🌚"
        ]
    @classmethod
    def create_spinner(cls, index, speed=1):
        """ Return a spinner object of the selected style.
        """
        if index >= len(cls.spinners):
            index = -1
        elif index < -len(cls.spinners):
            index = 0
        return Spinner(cls.spinners[index], speed)

    def __init__(self, spinner_str, speed):
        self._glyphs = list(spinner_str.decode('utf-8'))
        self._next_index = -1
        self._repitition = 0
        self._speed = speed

    def next(self):
        """ Return the next spinner glyph.
        """
        self._repitition += 1
        if self._repitition == self._speed:
            self._repitition = 0
            self._next_index = (self._next_index + 1) % len(self._glyphs)
        return self._glyphs[self._next_index]
