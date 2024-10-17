#!/bin/bash

# Путь к папке с файлами
folder_path="/u/gkozlov/cbm/STAR/2024_jul/git_vers/TPCCATracker"

# Имя файла, который нужно исключить
excluded_file="AliHLTTPCCADef.h"

# Выполняем замены для всех файлов в папке, кроме указанного файла
for file in "$folder_path"/*; do
  # Проверяем, что это файл, а не папка, и исключаем NoCheckFile.h
  if [[ -f "$file" && $(basename "$file") != "$excluded_file" ]]; then
    # Выполняем замены в файле
    sed -i 's/uint_v::SimdLen/SimdSizeInt/g' "$file"
    sed -i 's/int_v::SimdLen/SimdSizeInt/g' "$file"
    sed -i 's/float_v::SimdLen/SimdSizeFloat/g' "$file"
    
    echo "Done: $(basename "$file")"
  fi
done

echo "All is done."

