'use client';
import { useState } from 'react';
import { useParams, useRouter } from 'next/navigation';
import Navbar from '@/app/components/Navbar';
import Footer from '@/app/components/Footer';
import toast from 'react-hot-toast';

export default function ApplyJobPage() {
  const router = useRouter();
  const { id } = useParams(); // Job ID

  const [form, setForm] = useState({
    name: '',
    email: '',
    phone: '',
    resume: null as File | null,
    coverLetter: '',
  });

  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    setForm({ ...form, [e.target.name]: e.target.value });
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file && file.type === 'application/pdf') {
      setForm({ ...form, resume: file });
    } else {
      setError('Only PDF files are allowed.');
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setSubmitting(true);

    const { name, email, resume } = form;
    if (!name || !email || !resume) {
      setError('Please fill in all required fields.');
      setSubmitting(false);
      return;
    }

    try {
      // 🔼 Upload PDF to Cloudinary
      const uploadData = new FormData();
      uploadData.append('file', resume);
      uploadData.append('upload_preset', process.env.NEXT_PUBLIC_CLOUDINARY_UPLOAD_PRESET!);
      uploadData.append('folder', 'resumes');

      const cloudRes = await fetch(
        `https://api.cloudinary.com/v1_1/${process.env.NEXT_PUBLIC_CLOUDINARY_CLOUD_NAME}/upload`,
        {
          method: 'POST',
          body: uploadData,
        }
      );

      const cloudJson = await cloudRes.json();
      const resume_url = cloudJson.secure_url;
      if (!resume_url) throw new Error('Resume upload failed');

      // 📬 Submit application to backend
      const res = await fetch('https://95461b4a48b7.ngrok-free.app/apply', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          job_id: id,
          full_name: name,
          email: form.email,
          phone: form.phone,
          cover_letter: form.coverLetter,
          resume_url,
        }),
      });

      toast.success('Application submitted successfully!');
      router.push(`/find-jobs`);

      if (!res.ok) {
        const result = await res.json();
        toast.error('Upload failed.');
        throw new Error(result.error || 'Application failed');
      }

      router.push(`/find-jobs`);
    } catch (err: any) {
      setError(err.message || 'Something went wrong.');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#f9fafb] flex flex-col">
      <Navbar showBackButton={true} showActionButton={false} />

      <main className="flex-grow pt-24 pb-20 px-6 max-w-3xl mx-auto w-full">
        <h1 className="text-2xl font-bold text-gray-800 mb-2">Application Details</h1>
        <p className="text-sm text-gray-500 mb-8">Please fill out your information and upload your resume</p>

        <form onSubmit={handleSubmit} className="space-y-6 bg-white p-6 rounded-xl shadow-md">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
            <div>
              <label className="block font-medium text-sm text-gray-700 mb-1">Full Name *</label>
              <input type="text" name="name" value={form.name} onChange={handleChange} required className="w-full border border-gray-300 rounded-lg px-4 py-2" />
            </div>
            <div>
              <label className="block font-medium text-sm text-gray-700 mb-1">Email Address *</label>
              <input type="email" name="email" value={form.email} onChange={handleChange} required className="w-full border border-gray-300 rounded-lg px-4 py-2" />
            </div>
          </div>

          <div>
            <label className="block font-medium text-sm text-gray-700 mb-1">Phone Number</label>
            <input type="text" name="phone" value={form.phone} onChange={handleChange} className="w-full text-gray-800 border border-gray-300 rounded-lg px-4 py-2" placeholder="Optional" />
          </div>

          <div>
            <label className="block font-medium text-sm text-gray-700 mb-1">Resume * (PDF only)</label>
            <div className="border text-gray-400 border-dashed border-gray-300 rounded-lg px-4 py-6 text-center bg-gray-50">
              <input type="file" accept="application/pdf" onChange={handleFileChange} />
              {form.resume && (
                <p className="text-sm mt-2 text-purple-600">📄 {form.resume.name}</p>
              )}
            </div>
          </div>

          <div>
            <label className="block font-medium text-sm text-gray-700 mb-1">Cover Letter (Optional)</label>
            <textarea name="coverLetter" value={form.coverLetter} onChange={handleChange} className="w-full border text-gray-800 border-gray-300 rounded-lg px-4 py-2 min-h-[100px]" placeholder="Why should we hire you?" />
          </div>

          {error && <p className="text-red-500 text-sm">{error}</p>}

          <div className="flex justify-end gap-4">
            <button type="button" onClick={() => router.back()} className="px-4 py-2 text-sm rounded-lg bg-gray-100 hover:bg-gray-200 text-gray-700 font-medium">
              Cancel
            </button>
            <button type="submit" disabled={submitting} className="px-6 py-2 text-sm rounded-lg bg-gradient-to-tr from-[#7B61FF] to-[#9F80FF] text-white font-medium hover:opacity-90 transition">
              {submitting ? 'Submitting...' : 'Submit Application'}
            </button>
          </div>
        </form>
      </main>

      <Footer />
    </div>
  );
}