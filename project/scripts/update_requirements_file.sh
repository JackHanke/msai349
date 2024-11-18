filename='requirements.txt'
if [ -e "$filename" ]; then
    echo "Found existing $filename..."
    rm "$filename"  # Use quotes for safety
fi
echo "Creating new $filename..."
pip freeze > "$filename"  
