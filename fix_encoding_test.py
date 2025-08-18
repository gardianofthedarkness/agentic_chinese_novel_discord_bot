#!/usr/bin/env python3
"""
Fix encoding and test character extraction with corrupted text
"""

import re

def try_fix_encoding(text):
    """Try to fix garbled Chinese text encoding"""
    
    # Try different encoding fixes
    fixes = []
    
    # Try GBK to UTF-8
    try:
        if isinstance(text, str):
            # Convert to bytes first, then decode properly
            fixed = text.encode('latin1').decode('gbk')
            fixes.append(('gbk', fixed))
    except:
        pass
    
    # Try CP936
    try:
        if isinstance(text, str):
            fixed = text.encode('latin1').decode('cp936')
            fixes.append(('cp936', fixed))
    except:
        pass
    
    # Try Windows-1252 to UTF-8
    try:
        if isinstance(text, str):
            fixed = text.encode('windows-1252').decode('utf-8')
            fixes.append(('windows-1252', fixed))
    except:
        pass
    
    return fixes

# Test with the garbled text from Qdrant
garbled_text = "ħ������Ŀ¼ ��һ�� ����������ѧ԰�����е�һ����������ӵ����Ϊ'����ɱ��'����������������"

print("Original garbled text:")
print(garbled_text)
print()

print("Attempting encoding fixes:")
fixes = try_fix_encoding(garbled_text)

for encoding, fixed_text in fixes:
    print(f"Using {encoding}:")
    print(fixed_text[:100] + "...")
    print()

# Let's also try some manual character pattern matching on the garbled text
print("Testing character patterns on garbled text:")

# Pattern for potential names in garbled text
garbled_patterns = [
    r'[^\s]{2,4}(?=��|˵|��|��|ǰ|Ҫ|��|��)',  # Characters before common verbs
    r'��[^\s]{1,3}',  # Characters starting with common prefix
    r'[^\s]{2,4}��',   # Characters ending with common suffix
]

for pattern in garbled_patterns:
    matches = re.findall(pattern, garbled_text)
    if matches:
        print(f"Pattern {pattern}: {matches}")

# Test with some manual decode
print("\nTrying manual fixes:")

# Try interpreting as different encodings
test_chars = "��������"  # This looks like it might be a name
print(f"Test chars: {test_chars}")

# Try reverse encoding fixes
try:
    # Maybe it's double-encoded
    decoded = test_chars.encode('latin1').decode('gbk', errors='ignore')
    print(f"GBK decode: {decoded}")
except:
    print("GBK decode failed")

try:
    decoded = test_chars.encode('latin1').decode('cp936', errors='ignore')
    print(f"CP936 decode: {decoded}")
except:
    print("CP936 decode failed")