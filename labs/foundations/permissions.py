"""Compute Unix mode bits without changing real permissions."""
def symbolic(mode: int) -> str:
    if not isinstance(mode, int) or isinstance(mode, bool) or not 0 <= mode <= 0o777:
        raise ValueError("expected ordinary permission bits between 000 and 777")
    return "".join(letter if mode & bit else "-" for bit, letter in
                   [(256,"r"),(128,"w"),(64,"x"),(32,"r"),(16,"w"),(8,"x"),(4,"r"),(2,"w"),(1,"x")])

def apply_umask(requested: int, mask: int) -> int:
    return requested & ~mask & 0o777

if __name__ == "__main__":
    print("640:", symbolic(0o640))
    print("new file:", oct(apply_umask(0o666, 0o022)))
    print("new directory:", oct(apply_umask(0o777, 0o022)))
