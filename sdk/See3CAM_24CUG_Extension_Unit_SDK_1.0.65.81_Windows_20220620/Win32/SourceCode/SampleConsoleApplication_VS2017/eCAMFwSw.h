// The following ifdef block is the standard way of creating macros which make exporting 
// from a DLL simpler. All files within this DLL are compiled with the ECAMFWSW_EXPORTS
// symbol defined on the command line. this symbol should not be defined on any project
// that uses this DLL. This way any other project whose source files include this file see 
// ECAMFWSW_API functions as being imported from a DLL, whereas this DLL sees symbols
// defined with this macro as being exported.
#ifdef ECAMFWSW_EXPORTS
#define ECAMFWSW_API __declspec(dllexport)
#else
#define ECAMFWSW_API __declspec(dllimport)
#endif

BOOL GetDevicesCount(UINT32 *Cnt);

BOOL GetDeviceName(TCHAR *Name);

BOOL GetDevicePaths(TCHAR **DevicePaths);

BOOL InitExtensionUnit(TCHAR *USBInstanceID);

BOOL DeinitExtensionUnit();

BOOL ReadFirmwareVersion(UINT8 *pMajorVersion, UINT8 *pMinorVersion1, UINT16 *pMinorVersion2, UINT16 *pMinorVersion3);

BOOL GetCameraUniqueID(TCHAR *szUniqueID);

BOOL GetSpecialEffects24CUG(UINT8 *uEffectsMode);

BOOL SetSpecialEffects24CUG(UINT8 uEffectsMode);

BOOL GetDenoiseValue24CUG(UINT8 *uDenoiseValue);

BOOL SetDenoiseValue24CUG(UINT8 uDenoiseValue);

BOOL GetExpRoiMode24CUG(UINT8 *uExpRoiMode, UINT8 *uXPos, UINT8 *uYPos, UINT8 *uWindowSize);

BOOL SetExpRoiMode24CUG(UINT8 uExpRoiMode, UINT8 uXPos, UINT8 uYPos, UINT8 uWindowSize);

BOOL GetQFactor24CUG(UINT8 *uQFactorVal);

BOOL SetQFactor24CUG(UINT8 uQFactorVal);

BOOL RestoreDefault24CUG();

BOOL GetFlipMode24CUG(UINT8 *uFlipMode);

BOOL SetFlipMode24CUG(UINT8 uFlipMode);

BOOL GetFaceDetectionRect24CUG(UINT8 *uState, UINT8 *uStatusStructState, UINT8 *uOverlayRectState);

BOOL SetFaceDetectionRect24CUG(UINT8 uState, UINT8 uStatusStructState, UINT8 uOverlayRectState);

BOOL GetExposureCompensation24CUG(UINT32 *iExposureComp);

BOOL SetExposureCompensation24CUG(UINT32 iExposureComp);

BOOL GetFrameRateValue24CUG(UINT8 *uFrameRateVal);

BOOL SetFrameRateValue24CUG(UINT8 uFrameRateVal);

BOOL GetStrobeMode24CUG(UINT8 *uFlashValue);

BOOL SetStrobeMode24CUG(UINT8 uFlashValue);

BOOL GetAntiFlickerMode24CUG(UINT8 *iFlickerMode);

BOOL SetAntiFlickerMode24CUG(UINT8 iFlickerMode);

BOOL SetStreamMode24CUG(UINT8 iStreamMode, UINT8 iAutoFunctionsLock);

BOOL GetStreamMode24CUG(UINT8 *iStreamMode, UINT8 *iAutoFunctionsLock);

BOOL ResetDevice();
