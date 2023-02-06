def save_csv_log(head, value, is_create=False, file_name="test"):
    if len(value.shape) < 2:
        value = np.expand_dims(value, axis=0)
    df = pd.DataFrame(value)
    file_path = f"{output_dir}/{file_name}.csv"
    if not os.path.exists(file_path) or is_create:
        df.to_csv(file_path, header=head, index=False)
    else:
        with open(file_path, "a") as f:
            df.to_csv(f, header=False, index=False)
