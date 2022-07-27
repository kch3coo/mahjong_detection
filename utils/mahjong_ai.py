
from tabnanny import check

tiles_map = {1: 'bam-1', 2: 'bam-2', 3: 'bam-3', 4: 'bam-4', 5: 'bam-5', 6: 'bam-6', 7: 'bam-7', 8: 'bam-8', 9: 'bam-9', 10: 'char-1', 11: 'char-2', 12: 'char-3', 13: 'char-4', 14: 'char-5', 15: 'char-6', 16: 'char-7', 17: 'char-8', 18: 'char-9', 19: 'dots-1', 20: 'dots-2', 21: 'dots-3', 22: 'dots-4', 23: 'dots-5', 24: 'dots-6', 25: 'dots-7', 26: 'dots-8', 27: 'dots-9', 28: 'h-east', 29: 'h-green', 30: 'h-north', 31: 'h-red', 32: 'h-south', 33: 'h-west', 34: 'h-white'}

def tile_combo(tile_i, tile_j, tile_k):
    # return true if the three tiles form a combo
    # check if they are all the same
    if tile_i == tile_j and tile_j == tile_k:
        return 1
    # check if tiles are consecutive
    elif tile_i + 1 == tile_j and tile_j + 1 == tile_k:
        # check if tiles are in range 1 - 9, or 10 - 18, or 19 - 27
        if tile_k < 10 or (tile_i >= 10 and tile_k < 19) or (tile_i >= 19 and tile_k < 28):
            return 1
    return 0

def get_max_combo(tiles):
    memo = []
    sorted_Tiles = sorted(tiles)
    for i in range(len(tiles)):
        if i < 2:
            memo.append(0)
        else:
            memo.append(max(0, memo[i - 1], memo[i - 3] + tile_combo(sorted_Tiles[i - 2], sorted_Tiles[i - 1], sorted_Tiles[i])))
    # print(memo)
    result = []
    # iterate through memo, and add three tiles to result when memo[i] > memo[i - 1]
    rest = sorted_Tiles.copy()
    for i in range(len(memo)):
        if memo[i] > memo[i - 1]:
            result.append([sorted_Tiles[i - 2], sorted_Tiles[i - 1], sorted_Tiles[i]])
            rest.remove(sorted_Tiles[i - 2])
            rest.remove(sorted_Tiles[i - 1])
            rest.remove(sorted_Tiles[i])
    result.append(rest)
    print(result)
    return result
        



    
def get_pairs(tiles):
    pairs = []
    for i in range(len(tiles) - 1):
        if tiles[i] == tiles[i + 1]:
            pairs.append(tiles[i])
            pairs.append(tiles[i + 1])
    print(pairs)
    return pairs
                    




if __name__ == '__main__':
    print("configuring mahjong ai")
    get_pairs([2, 2, 4, 5])
    get_pairs([2, 4, 4, 5])
    get_pairs([2, 4, 5, 5])