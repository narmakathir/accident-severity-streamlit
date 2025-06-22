# --- Admin Page ---
elif page == "Admin":
    st.title("Admin Dashboard")
    
    # Simple password protection
    password = st.text_input("Enter Admin Password:", type="password")
    
    if password != "admin1":
        st.error("Incorrect password. Access denied.")
        st.stop()  # This stops execution if password is wrong
    
    st.warning("You are in admin mode. Changes here will affect all users.")
    
    st.subheader("Upload New Dataset")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        st.info("File uploaded successfully. Click the button below to update the system.")
        if st.button("Update System with New Dataset"):
            with st.spinner("Processing new dataset and retraining models..."):
                handle_dataset_upload(uploaded_file)
    
    st.subheader("Current System Information")
    st.write(f"Current target variable: {st.session_state.target_col}")
    st.write(f"Number of features: {len(st.session_state.X.columns) if st.session_state.X is not None else 0}")
    st.write(f"Number of models: {len(st.session_state.models)}")
    
    st.subheader("Reset to Default Dataset")
    if st.button("Reset System"):
        with st.spinner("Resetting to default dataset..."):
            df, label_encoders, target_col = load_default_data()
            X, y, X_train, X_test, y_train, y_test = prepare_model_data(df, target_col)
            models, scores_df = train_models(X_train, y_train, X_test, y_test)
            
            st.session_state.current_df = df
            st.session_state.label_encoders = label_encoders
            st.session_state.models = models
            st.session_state.scores_df = scores_df
            st.session_state.X = X
            st.session_state.y = y
            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
            st.session_state.target_col = target_col
            
            st.success("System reset to default dataset!")
