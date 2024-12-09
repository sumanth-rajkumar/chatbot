from django.shortcuts import render
from .models import UploadedFile
from .utils import (
    extract_text_from_ppt,
    extract_text_from_pdf,
    preprocess_and_store_chunks,
    create_faiss_index,
    save_faiss_index_and_chunks,
    load_faiss_index_and_chunks,
)

def upload_ppt(request):
    message = None
    extracted_text = None
    question = None
    retrieved_chunks = None
    # pylint: disable=no-member
    uploaded_files = UploadedFile.objects.all()  # Retrieve all uploaded files for the dropdown

    if request.method == 'POST':
        # Handle file upload
        if 'file' in request.FILES:
            uploaded_file = request.FILES['file']
            file_name = uploaded_file.name.lower()

            # Determine file type
            if file_name.endswith('.pptx'):
                file_type = 'pptx'
                extracted_text = extract_text_from_ppt(uploaded_file.file)
            elif file_name.endswith('.pdf'):
                file_type = 'pdf'
                extracted_text = extract_text_from_pdf(uploaded_file.file)
            else:
                message = "Unsupported file type. Please upload a PowerPoint or PDF file."
                return render(
                    request,
                    'file_manager/upload.html',
                    {
                        'message': message,
                        'uploaded_files': uploaded_files,
                        'question': question,
                        'retrieved_chunks': retrieved_chunks,
                    },
                )

            # Save file information in the database
            # pylint: disable=no-member
            uploaded = UploadedFile.objects.create(
                name=uploaded_file.name, file=uploaded_file, file_type=file_type
            )
            uploaded.extracted_text = extracted_text

            # Process text: chunking, embeddings, and indexing
            chunks, embeddings = preprocess_and_store_chunks(extracted_text)
            faiss_index = create_faiss_index(embeddings)
            save_faiss_index_and_chunks(faiss_index, chunks, uploaded.name)

            # Update database with FAISS index and chunks paths
            uploaded.faiss_index_path = f"{uploaded.name}_faiss.index"
            uploaded.chunks_path = f"{uploaded.name}_chunks.pkl"
            uploaded.save()

            message = f"{file_type.upper()} file uploaded, processed, and indexed successfully!"

        # Handle question submission
        elif 'question' in request.POST:
            question = request.POST['question']
            file_selection = request.POST.get('file_selection')  # Get selected file

            if file_selection == "all":
                # Combine content from all files
                chunks = []
                # pylint: disable=no-member
                for file in UploadedFile.objects.all():
                    index, file_chunks = load_faiss_index_and_chunks(file.name)
                    chunks.extend(file_chunks)
                    # Note: Retrieving relevant chunks for testing
                    relevant_chunk_ids = index.search(
                        preprocess_and_store_chunks(question)[1], k=3
                    )[1].flatten().tolist()
                    retrieved_chunks = [chunks[i] for i in relevant_chunk_ids]
            else:
                # Get content from a specific file
                # pylint: disable=no-member
                selected_file = UploadedFile.objects.filter(id=file_selection).first()
                if selected_file:
                    index, chunks = load_faiss_index_and_chunks(selected_file.name)
                    relevant_chunk_ids = index.search(
                        preprocess_and_store_chunks(question)[1], k=3
                    )[1].flatten().tolist()
                    retrieved_chunks = [chunks[i] for i in relevant_chunk_ids]
                else:
                    message = "No context available for the selected file(s). Please upload a file first."

      # Pass project names to the template
    project_names = [file.name for file in uploaded_files]

    return render(
        request,
        'file_manager/upload.html',
        {
            'message': message,
            'uploaded_files': uploaded_files,
            'question': question,
            'retrieved_chunks': retrieved_chunks,
            'project_names': project_names,  # Pass project names to template
        },
    )



import json
from django.http import JsonResponse
from .models import UploadedFile
from .utils import (
    preprocess_text,
    load_faiss_index_and_chunks,
    find_relevant_chunks,
)

def ask_question(request):
    """
    Handle chatbot question queries based on the selected project.
    """
    if request.method == 'POST':
        try:
            # Parse the question and project name from the request
            data = json.loads(request.body)
            question = data.get('question', '').strip()
            project = data.get('project', '').strip()

            if not question:
                return JsonResponse({'error': 'No question provided.'}, status=400)

            if not project:
                return JsonResponse({'error': 'No project selected.'}, status=400)

            # Load the FAISS index and chunks for the selected project
            # pylint: disable=no-member
            selected_file = UploadedFile.objects.filter(name=project).first()
            if not selected_file:
                return JsonResponse({'error': 'Selected project not found.'}, status=404)

            index, chunks = load_faiss_index_and_chunks(selected_file.name)

            # Find the most relevant chunks for the question
            relevant_chunks = find_relevant_chunks(question, index, chunks)

            # Combine retrieved chunks into a single response
            if relevant_chunks:
                combined_response = " ".join(relevant_chunks)
            else:
                combined_response = "No relevant information found for your question."

            # Return the response as JSON
            return JsonResponse({'answer': combined_response}, status=200)

        except Exception as e:
            return JsonResponse({'error': f"An error occurred: {e}"}, status=500)
    else:
        return JsonResponse({'error': 'Invalid request method.'}, status=405)





         # pylint: disable=no-member
