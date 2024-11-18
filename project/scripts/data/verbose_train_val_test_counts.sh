cd data
for dir in */; do
    if [ -d "$dir" ]; then
        count=$(find "$dir" -type f | wc -l)
        echo "Folder: $dir contains $count files."
    fi
done

