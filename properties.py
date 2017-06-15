# -*- coding: utf-8 -*- 
dict_en_vi="dict.en-vi.txt"
train_en="train.en.txt"
train_vi="train.vi.txt"
vocab_en="vocab.en.txt"
vocab_vi="vocab.vi.txt"
vi_char=['a','ă','â','b','c',
        'd','đ','e','ê','f',
        'g','h','i','j','k',
        'l','m','n','o','ô',
        'ơ','p','q','r','s',
        't','u','ư','v','w',
        'x','y','z',
        'à','á','ả','ã','ạ',
        'ằ','ắ','ẳ','ẵ','ặ',
        'ầ','ấ','ẩ','ẫ','ậ',
        'ì','í','ỉ','ĩ','ị',
        'ò','ó','ỏ','õ','ọ',
        'ồ','ố','ổ','ỗ','ộ',
        'ờ','ớ','ở','ỡ','ợ',
        'è','é','ẻ','ẽ','ẹ',
        'ề','ế','ể','ễ','ệ',
        'ù','ú','ủ','ũ','ụ',
        'ừ','ứ','ử','ữ','ự',
        'ỳ','ý','ỷ','ỹ','ỵ']
en_char=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p',
        'q','r','s','t','u','v','w','x','y','z']
num_char=['0','1','2','3','4','5','6','7','8','9']
spec_char=['!','"','#','$','%','&',"'",'(',')','*','+',',','-','.','/',':',
            ';','<','=','>','?','@','[','\\',']','^','_','`','{','|','}','~']
max_len=50  # sentences maximum length
features_size=50  # features size of word vector
batch_size=128
epochs=11