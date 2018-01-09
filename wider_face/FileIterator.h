#ifndef DirIterator_h__
#define DirIterator_h__

#include <string>
#include <io.h>

class CFileIterator
{
public:
    CFileIterator(const std::string& path, const std::string filter = "*.*");
    ~CFileIterator();

    bool FindNext();
    std::string FileName() const;
    std::string FullFileName() const;

private:
    std::string m_sPath;
    std::string m_sFilter;

    struct _finddata_t m_fileAttribute;
    long long m_hFile;

    bool m_bFirst;
};
#endif // DirIterator_h__